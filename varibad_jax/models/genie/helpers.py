import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
import numpy as np
from varibad_jax.models.genie.lam import LatentActionModel


@hk.transform_with_state
def lam_apply_fn(config: FrozenConfigDict, obs: jnp.ndarray, **kwargs):
    model = LatentActionModel(config=config)
    return model(obs, **kwargs)


def init_params_lam(
    config: FrozenConfigDict,
    rng: PRNGKey,
    observation_space: tuple,
):
    # [B, T, D]
    dummy_obs = jnp.zeros((1, 1, *observation_space.shape))
    params, state = lam_apply_fn.init(
        rng,
        config=config,
        obs=dummy_obs,
    )
    return params, state
