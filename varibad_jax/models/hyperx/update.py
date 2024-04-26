from functools import partial
import time
from typing import Optional, Callable

from absl import logging
import einops
from flax.training import train_state
from flax.training.train_state import TrainState
import gym
import haiku as hk
import jax
import optax
import jax.numpy as jnp
from jax.random import PRNGKey
from ml_collections.config_dict import FrozenConfigDict
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from varibad_jax.models.hyperx.rnd import RND
from varibad_jax.models.base import BaseModel

tfd = tfp.distributions
tfb = tfp.bijectors


def compute_hyperx_bonus(predictor_params, ts_prior, ts_predictor, rng, obs, latent):
    prior_key, predictor_key = jax.random.split(rng)
    prior_output, _ = ts_prior.apply_fn(
        ts_prior.params, None, prior_key, obs=obs, latent=latent
    )
    predictor_output, state = ts_predictor.apply_fn(
        predictor_params, None, predictor_key, obs=obs, latent=latent
    )

    hyperstate_bonus = optax.squared_error(prior_output, predictor_output)

    if obs.ndim == 3:
        hyperstate_bonus = einops.reduce(hyperstate_bonus, "b t d -> b t", "mean")
    elif obs.ndim == 2:
        hyperstate_bonus = einops.reduce(hyperstate_bonus, "b d -> b", "mean")

    return hyperstate_bonus, state


class HyperXBonuses(BaseModel):

    @hk.transform_with_state
    def model(self, **kwargs):
        model = RND(config=self.config)
        return model(**kwargs)

    def _init_model(self):
        # [B, T, D]
        dummy_obs = jnp.zeros((1, 1, *self.observation_shape))
        dummy_latent = jnp.zeros((1, 1, self.config.latent_dim * 2))

        logging.info(f"dummy obs shape: {dummy_obs.shape}")
        logging.info(f"dummy latent shape: {dummy_latent.shape}")

        self._params, self._state = self.model.init(
            self._key,
            obs=dummy_obs,
            latent=dummy_latent,
        )

    def loss_fn(
        self,
        predictor_params,
        ts_predictor: TrainState,
        ts_prior: TrainState,
        ts_vae: TrainState,
        state: hk.State,
        vae_state: hk.State,
        config: FrozenConfigDict,
        batch,
        rng_key: PRNGKey,
    ):
        logging.debug("inside compute_hyperx_loss")
        T, B, *_ = batch.next_obs.shape

        encode_key, prior_key, hyperx_key = jax.random.split(rng_key, 3)

        # encode the first state to get prior
        prior_output = get_prior_fn(ts_vae.params, prior_key, batch_size=B)
        hidden_state = prior_output.hidden_state

        # encode the full trajectory and get latent posteriors
        encode_outputs, state = ts_vae.apply_fn(
            ts_vae.params,
            vae_state,
            encode_key,
            states=batch.next_obs,
            actions=batch.actions,
            rewards=batch.rewards,
            hidden_state=hidden_state,
            mask=jnp.ones((T, B)) if config.vae.encoder == "transformer" else None,
        )

        latent_mean = encode_outputs.latent_mean
        latent_logvar = encode_outputs.latent_logvar
        latent_belief = jnp.concatenate([latent_mean, latent_logvar], axis=-1)

        hyperstate_bonus, state = compute_hyperx_bonus(
            predictor_params,
            ts_prior,
            ts_predictor,
            rng=hyperx_key,
            obs=batch.next_obs,
            latent=latent_belief,
        )

        loss = jnp.mean(hyperstate_bonus)
        return loss, state
