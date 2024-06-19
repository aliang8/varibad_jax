from absl import logging
from typing import Tuple, NamedTuple
import json
import jax
import chex
import optax
import pickle
import haiku as hk
import jax.numpy as jnp
from pathlib import Path
from functools import partial
from ml_collections.config_dict import ConfigDict
from varibad_jax.agents.common import PolicyOutput


@optax.inject_hyperparams
def optimizer(lr, eps):
    return optax.chain(
        optax.clip_by_global_norm(2.0),
        optax.adamw(lr, eps=eps),
    )


@chex.dataclass
class TrainingState:
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState


class BaseModel:
    def __init__(
        self,
        config,
        key: jax.random.PRNGKey,
        observation_shape: Tuple,
        action_dim: int,
        input_action_dim: int,
        continuous_actions: bool,
        task_dim: int = 0,
        model_key: str = "",
        load_from_ckpt: bool = False,
        ckpt_file: str = "",
        ckpt_dir: str = "",
        num_devices: int = 1,
    ):
        self.config = config
        self.is_continuous = continuous_actions
        self.action_dim = action_dim
        self.observation_shape = observation_shape
        self.input_action_dim = input_action_dim
        self.task_dim = task_dim
        self.num_devices = num_devices

        self._init_key = key
        self._ts = TrainingState(params=None, state=None, opt_state=None)

        w_init = hk.initializers.VarianceScaling(scale=2.0)
        b_init = hk.initializers.Constant(0.0)
        self.init_kwargs = dict(w_init=w_init, b_init=b_init)

        if load_from_ckpt:
            if ckpt_dir:
                ckpt_dir = Path(ckpt_dir) / "model_ckpts"
                # search and sort by epoch
                ckpt_files = sorted(ckpt_dir.glob("*.pkl"), reverse=True)
                ckpt_file = ckpt_files[0]
            assert ckpt_file, "ckpt_file not provided"
            logging.info(f"loading {config.name} from checkpoint {ckpt_file}")
            with open(ckpt_file, "rb") as f:
                ckpt = pickle.load(f)
                if model_key:
                    params, state = (
                        ckpt[model_key]["params"],
                        ckpt[model_key]["state"],
                    )
                else:
                    params, state = ckpt["params"], ckpt["state"]

            # print(params["LatentActionIDM/~/image_encoder/downsampling_block/conv2_d"]["w"].sum())
            # import ipdb

            # ipdb.set_trace()
            # # get first of params, since we are using pmap TODO: fix this
            # params = jax.tree_util.tree_map(lambda x: x[0], params)
            # state = jax.tree_util.tree_map(lambda x: x[0], state)
        else:
            # replicate init_key (same initial weights on all devices)
            self._init_key = jnp.tile(self._init_key[None], (num_devices, 1))
            params, state = self._init_model(self._init_key)

        num_params = (
            sum(p.size for p in jax.tree_util.tree_leaves(params)) / num_devices
        )
        logging.info(f"number of {config.name} parameters: {num_params}")

        opt_state = self._init_opt(params)
        self._ts = TrainingState(params=params, state=state, opt_state=opt_state)

    @partial(jax.pmap, axis_name="device", static_broadcasted_argnums=(0,))
    def _init_model(self, init_key: jax.random.PRNGKey):
        raise NotImplementedError

    @partial(jax.pmap, axis_name="device", static_broadcasted_argnums=(0,))
    def _init_opt(self, params) -> None:
        """Initialize Adam optimizer for training."""
        if self.config.use_lr_scheduler:
            # apply linear warmup
            lr = optax.linear_schedule(
                init_value=0,
                end_value=self.config.lr,
                transition_steps=self.config.warmup_steps,
            )
        else:
            lr = self.config.lr
        self.opt = optimizer(lr, self.config.eps)
        opt_state = self.opt.init(params)
        return opt_state

    @partial(jax.pmap, axis_name="device", static_broadcasted_argnums=(0, 4))
    def update_model(self, ts, rng, batch, update_model):
        if update_model:
            logging.info("updating model")

        (loss, (metrics, extra, new_state)), grads = jax.value_and_grad(
            self.loss_fn, has_aux=True
        )(ts.params, ts.state, rng, batch, is_training=update_model)

        # sync gradients
        grads = jax.lax.pmean(grads, "device")

        if update_model:
            grads, new_opt_state = self.opt.update(grads, ts.opt_state, ts.params)
            new_params = optax.apply_updates(ts.params, grads)
            new_ts = TrainingState(
                params=new_params, state=new_state, opt_state=new_opt_state
            )
        else:
            new_ts = ts

        # sync metrics
        metrics = jax.lax.pmean(metrics, "device")
        return new_ts, metrics, extra

    def update(self, rng, batch, update_model=True):
        rng = jax.random.split(rng, self.num_devices)
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((self.num_devices, -1, *x.shape[1:])), batch
        )

        self._ts, metrics, extra = self.update_model(self._ts, rng, batch, update_model)
        metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
        return metrics, extra

    @property
    def save_dict(self):
        params = self._ts.params
        state = self._ts.state
        if self.num_devices > 1:
            params = jax.tree_util.tree_map(lambda x: x[0], params)
            state = jax.tree_util.tree_map(lambda x: x[0], state)
        return dict(
            params=params,
            state=state,
        )


class BaseAgent(BaseModel):
    @partial(jax.jit, static_argnames=("self", "is_training"))
    def get_action_jit(self, ts, rng, env_state, **kwargs):
        # get first of params
        params = jax.tree_util.tree_map(lambda x: x[0], ts.params)
        state = jax.tree_util.tree_map(lambda x: x[0], ts.state)

        return self.model.apply(params, state, rng, self, env_state, **kwargs)

    def get_action(self, rng, env_state, **kwargs):
        return self.get_action_jit(self._ts, rng, env_state, **kwargs)


class RandomActionAgent(BaseAgent):

    @partial(jax.jit, static_argnames=("self", "is_training"))
    def get_action_jit(self, params, state, rng, env_state, **kwargs):
        logging.info("random action")
        logging.info(kwargs)
        if self.is_continuous:
            action = jax.random.uniform(rng, (env_state.shape[0], self.action_dim))
        else:
            action = jax.random.randint(rng, (env_state.shape[0],), 0, 3)

        return PolicyOutput(action=action, value=jnp.zeros_like(action)), None

    def _init_model(self):
        self._params, self._state = None, None
        return
