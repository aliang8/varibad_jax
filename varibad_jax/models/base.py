from absl import logging
from typing import Tuple
import json
import jax
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
        # optax.clip_by_global_norm(config.policy.max_grad_norm),
        optax.adamw(lr, eps=eps),
    )


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
    ):
        self.config = config
        self.is_continuous = continuous_actions
        self.action_dim = action_dim
        self.observation_shape = observation_shape
        self.input_action_dim = input_action_dim
        self.task_dim = task_dim

        self._init_key = key
        self._params = None
        self._state = None
        self._opt_state = None

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
                    self._params, self._state = (
                        ckpt[model_key]["params"],
                        ckpt[model_key]["state"],
                    )
                else:
                    self._params, self._state = ckpt["params"], ckpt["state"]
        else:
            self._init_model()
        num_params = sum(p.size for p in jax.tree_util.tree_leaves(self._params))
        logging.info(f"number of {config.name} parameters: {num_params}")

        self._init_opt()

    def _init_model(self):
        raise NotImplementedError

    def _init_opt(self) -> None:
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
        self._opt_state = self.opt.init(self._params)

    @partial(jax.jit, static_argnums=(0, 6))
    def update_model(self, params, state, opt_state, rng, batch, update_model):
        logging.info("updating model")
        (loss, (metrics, state)), grads = jax.value_and_grad(
            self.loss_fn, has_aux=True
        )(params, state, rng, batch)
        if update_model:
            grads, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, grads)
        else:
            opt_state = self._opt_state
        return params, state, opt_state, metrics

    def update(self, rng, batch, update_model=True):
        self._params, self._state, self._opt_state, metrics = self.update_model(
            self._params, self._state, self._opt_state, rng, batch, update_model
        )
        return metrics

    @property
    def save_dict(self):
        return dict(
            params=self._params,
            state=self._state,
        )


class BaseAgent(BaseModel):

    @partial(jax.jit, static_argnames=("self", "is_training"))
    def get_action_jit(self, params, state, rng, env_state, **kwargs):
        logging.info("inside get action")
        logging.info(kwargs)
        # jax.debug.print("{x}", x=params["ActorCritic/~/task_embed"]["w"].mean())
        return self.model.apply(params, state, rng, self, env_state, **kwargs)

    # using env_state because of naming conflict with hk state
    def get_action(self, rng, env_state, **kwargs):
        return self.get_action_jit(self._params, self._state, rng, env_state, **kwargs)


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
