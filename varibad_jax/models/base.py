from absl import logging
from typing import Tuple
import jax
import optax
import pickle
import haiku as hk
from pathlib import Path
from functools import partial
from ml_collections.config_dict import ConfigDict


@optax.inject_hyperparams
def optimizer(lr, eps):
    return optax.chain(
        # optax.clip_by_global_norm(config.policy.max_grad_norm),
        optax.adam(lr, eps=eps),
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
    ):
        self.config = config
        self.is_continuous = continuous_actions
        self.action_dim = action_dim
        self.observation_shape = observation_shape
        self.input_action_dim = input_action_dim
        self.task_dim = task_dim

        self._key = key
        self._params = None
        self._state = None
        self._opt_state = None

        if load_from_ckpt:
            logging.info(f"loading {config.name} from checkpoint {ckpt_file}")
            with open(ckpt_file, "rb") as f:
                ckpt = pickle.load(f)
                self._params, self._state = (
                    ckpt[model_key]["params"],
                    ckpt[model_key]["state"],
                )
        else:
            self._init_model()
        num_params = sum(p.size for p in jax.tree_util.tree_leaves(self._params))
        logging.info(f"number of {config.name} parameters: {num_params}")

        self._init_opt()

    def _init_model(self):
        raise NotImplementedError

    def _init_opt(self) -> None:
        """Initialize Adam optimizer for training."""
        if self.config.anneal_lr:
            lr = optax.polynomial_schedule(
                init_value=self._learning_rate_start,
                end_value=self._learning_rate_end,
                power=1,  # 1: Linear decay
                transition_steps=self._learning_rate_decay_steps,
            )
        else:
            lr = self.config.lr
        self.opt = optimizer(lr, self.config.eps)
        self._opt_state = self.opt.init(self._params)

    @partial(jax.jit, static_argnums=(0, 5))
    def update_model(self, params, state, rng, batch, update_model):
        logging.info("updating model")
        (loss, (metrics, state)), grads = jax.value_and_grad(
            self.loss_fn, has_aux=True
        )(params, state, rng, batch)
        if update_model:
            grads, opt_state = self.opt.update(grads, self._opt_state)
            params = optax.apply_updates(params, grads)
        return params, state, opt_state, metrics

    def update(self, rng, batch, update_model=True):
        self._params, self._state, self._opt_state, metrics = self.update_model(
            self._params, self._state, rng, batch, update_model
        )
        return metrics

    @property
    def save_dict(self):
        return dict(
            params=self._params,
            state=self._state,
        )


class BaseAgent(BaseModel):

    @partial(jax.jit, static_argnums=(0,))
    def get_action_jit(self, params, state, rng, env_state, **kwargs):
        logging.info("inside get action")
        # jax.debug.print("{x}", x=params["ActorCritic/~/task_embed"]["w"].mean())
        return self.model.apply(params, state, rng, self, env_state, **kwargs)

    # using env_state because of naming conflict with hk state
    def get_action(self, rng, env_state, **kwargs):
        return self.get_action_jit(self._params, self._state, rng, env_state, **kwargs)
