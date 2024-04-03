import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
import numpy as np
from varibad_jax.models.decision_transformer.model import DecisionTransformer


@hk.transform
def dt_fn(config: FrozenConfigDict, is_continuous: bool, action_dim: int, **kwargs):
    model = DecisionTransformer(
        config=config, is_continuous=is_continuous, action_dim=action_dim
    )
    return model(**kwargs)


def init_params_dt(
    config: FrozenConfigDict,
    rng_key: PRNGKey,
    observation_space: tuple,
    is_continuous: bool,
    input_action_dim: int,
    action_dim: int,
):
    t = 2
    bs = 2

    # import ipdb

    # ipdb.set_trace()
    dummy_states = np.zeros((t, bs, *observation_space.shape), dtype=np.float32)
    dummy_actions = np.zeros((t, bs, input_action_dim))
    dummy_rewards = np.zeros((t, bs, 1))
    dummy_mask = np.ones((t, bs))

    dt_params = dt_fn.init(
        rng_key,
        config=config,
        is_continuous=is_continuous,
        action_dim=action_dim,
        **dict(
            states=dummy_states,
            actions=dummy_actions,
            rewards=dummy_rewards,
            mask=dummy_mask,
        )
    )
    return dt_params
