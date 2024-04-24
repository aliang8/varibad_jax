from absl import logging
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
import numpy as np
from varibad_jax.models.hyperx.rnd import RND
import optax
import einops

@hk.transform_with_state
def hyperx_apply_fn(config: FrozenConfigDict, obs: jnp.ndarray, latent: jnp.ndarray, **kwargs):
    model = RND(config=config)
    return model(obs, latent, **kwargs)


def init_params_hyperx(
    config: FrozenConfigDict,
    rng: PRNGKey,
    observation_space: tuple,
):
    # [B, T, D]
    dummy_obs = jnp.zeros((1, 1, *observation_space.shape))
    dummy_latent = jnp.zeros((1, 1, config.latent_dim * 2))

    logging.info(f"dummy obs shape: {dummy_obs.shape}")
    logging.info(f"dummy latent shape: {dummy_latent.shape}")

    params, state = hyperx_apply_fn.init(
        rng,
        config=config.rnd,
        obs=dummy_obs,
        latent=dummy_latent,
    )
    return params, state

def compute_hyperx_bonus(predictor_params, ts_prior, ts_predictor, rng, obs, latent):
    prior_key, predictor_key = jax.random.split(rng)
    prior_output, _ = ts_prior.apply_fn(ts_prior.params, None, prior_key, obs=obs, latent=latent)
    predictor_output, state = ts_predictor.apply_fn(predictor_params, None, predictor_key, obs=obs, latent=latent)

    hyperstate_bonus = optax.squared_error(prior_output, predictor_output)
    
    if obs.ndim == 3:
        hyperstate_bonus = einops.reduce(hyperstate_bonus, 'b t d -> b t', 'mean')
    elif obs.ndim == 2:
        hyperstate_bonus = einops.reduce(hyperstate_bonus, 'b d -> b', 'mean')
    
    return hyperstate_bonus, state