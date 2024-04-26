from absl import logging
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
import numpy as np
import einops
import optax
from functools import partial
from tensorflow_probability.substrates import jax as tfp

from varibad_jax.models.base import BaseModel
from varibad_jax.models.varibad.vae import VaribadVAE

tfd = tfp.distributions
tfb = tfp.bijectors


@hk.transform_with_state
def encode_trajectory(config, **kwargs):
    model = VaribadVAE(config=config)
    return model.encode(**kwargs)


@hk.transform
def get_prior(config, **kwargs):
    model = VaribadVAE(config=config)
    return model.get_prior(**kwargs)


@hk.transform_with_state
def decode(config, **kwargs):
    model = VaribadVAE(config=config)
    return model.decode(**kwargs)


class VariBADModel(BaseModel):
    def _init_model(self):
        t, bs = 2, 2
        dummy_states = np.zeros((t, bs, *self.observation_shape), dtype=np.float32)
        dummy_actions = np.zeros((t, bs, self.input_action_dim))
        dummy_rewards = np.zeros((t, bs, 1))

        if self.config.encoder.name == "lstm":
            dummy_hs = np.zeros((bs, self.config.encoder.lstm_hidden_size))
        else:
            dummy_hs = None

        dummy_latents = np.zeros((t, bs, self.config.latent_dim))
        if self.config.encoder.name == "transformer":
            dummy_mask = np.ones((t, bs))
        else:
            dummy_mask = None

        encoder_key, decoder_key = jax.random.split(self._key)
        encoder_params, encoder_state = encode_trajectory.init(
            encoder_key,
            config=self.config,
            states=dummy_states,
            actions=dummy_actions,
            rewards=dummy_rewards,
            hidden_state=dummy_hs,
            mask=dummy_mask,
            is_training=True,
        )
        decoder_params, decoder_state = decode.init(
            decoder_key,
            config=self.config,
            latent_samples=dummy_latents,
            prev_states=dummy_states,
            next_states=dummy_states,
            actions=dummy_actions,
            is_training=True,
        )
        encoder_params.update(decoder_params)
        encoder_state.update(decoder_state)
        self._params = encoder_params
        self._state = encoder_state

    @partial(jax.jit, static_argnums=(0, 2))
    def get_prior(self, rng, batch_size):
        logging.info("inside get_prior")
        return get_prior.apply(
            self._params, rng, config=self.config, batch_size=batch_size
        )

    @partial(jax.jit, static_argnums=(0,))
    def encode_trajectory(self, rng, *args, **kwargs):
        logging.info("inside encode_trajectory")
        return encode_trajectory.apply(
            self._params, self._state, rng, config=self.config, *args, **kwargs
        )

    @partial(jax.jit, static_argnums=(0,))
    def decode(self, rng, *args, **kwargs):
        logging.info("inside decode")
        return decode.apply(
            self._params, self._state, rng, config=self.config, *args, **kwargs
        )

    def loss_fn(self, params: hk.Params, state: hk.State, rng_key: PRNGKey, batch):
        logging.debug("inside loss_vae")
        T, B, *_ = batch.next_obs.shape
        logging.debug(f"T: {T}, B: {B}")

        encode_key, prior_key, decode_key = jax.random.split(rng_key, 3)

        # encode the first state to get prior
        prior_output = self.get_prior(rng=prior_key, batch_size=B)
        hidden_state = prior_output.hidden_state
        prior_mean = jnp.expand_dims(prior_output.latent_mean, axis=0)
        prior_logvar = jnp.expand_dims(prior_output.latent_logvar, axis=0)
        prior_samples = jnp.expand_dims(prior_output.latent_sample, axis=0)

        # encode the full trajectory and get latent posteriors
        encode_outputs, state = encode_trajectory.apply(
            params,
            state,
            encode_key,
            config=self.config,
            states=batch.next_obs,
            actions=batch.actions,
            rewards=batch.rewards,
            hidden_state=hidden_state,
            mask=jnp.ones((T, B)) if self.config.encoder == "transformer" else None,
        )

        latent_mean = encode_outputs.latent_mean
        latent_logvar = encode_outputs.latent_logvar
        latent_samples = encode_outputs.latent_sample

        latent_mean = jnp.concatenate([prior_mean, latent_mean], axis=0)
        latent_logvar = jnp.concatenate([prior_logvar, latent_logvar], axis=0)
        latent_samples = jnp.concatenate([prior_samples, latent_samples], axis=0)

        num_elbos = latent_samples.shape[0]
        num_decodes = T

        # import ipdb

        # ipdb.set_trace()

        # reconstruction loss
        prev_obs = batch.prev_obs
        next_obs = batch.next_obs
        actions = batch.actions
        rewards = batch.rewards

        def repeat_elbos(x):
            # add an extra dim and repeat num elbos time
            return einops.repeat(x, "t b ... -> elbo t b ...", elbo=num_elbos)

        dec_prev_obs = repeat_elbos(prev_obs)
        dec_next_obs = repeat_elbos(next_obs)
        dec_actions = repeat_elbos(actions)
        dec_rewards = repeat_elbos(rewards)

        dec_embedding = einops.repeat(
            latent_samples, "elbo b d -> elbo t b d", t=num_decodes
        )

        logging.info(
            f"next_obs: {dec_next_obs.shape}, embedding: {dec_embedding.shape}"
        )

        # decode trajectory
        decode_outputs, state = decode.apply(
            params,
            state,
            decode_key,
            config=self.config,
            latent_samples=dec_embedding,
            prev_states=dec_prev_obs,
            next_states=dec_next_obs,
            actions=dec_actions,
        )

        # reward reconstruction loss
        # should be [num_elbos, num_decodes, B, 1]
        rew_pred = decode_outputs.rew_pred
        rew_recon_loss = optax.squared_error(rew_pred, dec_rewards)
        rew_recon_loss = rew_recon_loss.sum(axis=0).sum(axis=0).mean()

        # kl loss
        if self.config.kl_to_fixed_prior:
            posterior = tfd.Normal(loc=latent_mean, scale=jnp.exp(latent_logvar))
            prior = tfd.Normal(loc=0, scale=1)
        else:
            all_means = jnp.concatenate(
                (
                    jnp.zeros((1, *latent_mean.shape[1:])),
                    latent_mean,
                )
            )
            all_logvars = jnp.concatenate(
                (
                    jnp.zeros((1, *latent_logvar.shape[1:])),
                    latent_logvar,
                )
            )
            mu = all_means[1:]
            m = all_means[:-1]
            logE = all_logvars[1:]
            logS = all_logvars[:-1]
            posterior = tfd.Normal(loc=mu, scale=jnp.exp(logE))
            prior = tfd.Normal(loc=m, scale=jnp.exp(logS))

        kld = tfd.kl_divergence(posterior, prior).sum(axis=-1)
        kld = kld.sum(axis=0).sum(axis=0).mean()

        total_loss = (
            self.config.kl_weight * kld + self.config.rew_recon_weight * rew_recon_loss
        )

        loss_dict = {
            "kld": kld,
            "rew_recon_loss": rew_recon_loss,
            "total_loss": total_loss,
        }

        return total_loss, (loss_dict, state)
