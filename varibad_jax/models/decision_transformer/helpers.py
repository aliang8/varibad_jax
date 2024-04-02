import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
import numpy as np
from varibad_jax.models.transformer_encoder import DecisionTransformer


@hk.transform
def dt_fn(config: FrozenConfigDict, **kwargs):
    model = DecisionTransformer(config=config)
    return model.encode(**kwargs)


def init_params_dt(
    config: FrozenConfigDict,
    rng_key: PRNGKey,
    observation_space: tuple,
    action_dim: int,
):
    t = 2
    bs = 2
    dummy_states = np.zeros((t, bs, *observation_space.shape), dtype=np.float32)
    dummy_actions = np.zeros((t, bs, action_dim))
    dummy_rewards = np.zeros((t, bs, 1))

    dt_params = dt_fn.init(
        rng_key,
        config=config,
        **dict(
            states=dummy_states,
            actions=dummy_actions,
            rewards=dummy_rewards,
        )
    )
    return dt_params
