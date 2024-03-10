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
import varibad_jax.utils.general_utils as gutl
from varibad_jax.models.helpers import Batch, encode_trajectory, decode

tfd = tfp.distributions
tfb = tfp.bijectors


def loss_vae(
    params,
    ts: TrainState,
    rng_key: PRNGKey,
    batch: Batch,
    config: FrozenConfigDict,
    decode_fn: Callable,
):
    logging.debug("inside loss_vae")
    T, B, state_dim = batch.prev_obs.shape
    logging.debug(f"T: {T}, B: {B}, state_dim: {state_dim}")

    encode_key, sample_key, decode_key = jax.random.split(rng_key, 3)

    # encode the full trajectory and get latent posteriors
    encode_outputs = ts.apply_fn(
        params,
        encode_key,
        states=batch.prev_obs,
        actions=batch.actions,
        rewards=batch.rewards,
    )

    latents = encode_outputs.latent
    latent_mean = encode_outputs.latent_mean
    latent_logvar = encode_outputs.latent_logvar
    latent_dist = tfd.Normal(loc=latent_mean, scale=jnp.exp(0.5 * latent_logvar))
    latent_samples = latent_dist.sample(seed=sample_key)

    num_elbos = latents.shape[0]
    num_decodes = T

    # reconstruction loss
    prev_obs = batch.prev_obs
    next_obs = batch.next_obs
    actions = batch.actions
    rewards = batch.rewards

    def repeat_elbos(x):
        return einops.repeat(x, "t b d -> elbo t b d", elbo=num_elbos)

    dec_prev_obs = repeat_elbos(prev_obs)
    dec_next_obs = repeat_elbos(next_obs)
    dec_actions = repeat_elbos(actions)
    dec_rewards = repeat_elbos(rewards)

    dec_embedding = einops.repeat(
        latent_samples, "elbo b d -> elbo t b d", t=num_decodes
    )

    logging.info(f"next_obs: {dec_next_obs.shape}, embedding: {dec_embedding.shape}")

    # decode trajectory
    decode_outputs = decode_fn(
        params,
        decode_key,
        latent_samples=dec_embedding,
        next_states=dec_next_obs,
    )

    # reward reconstruction loss
    # should be [num_elbos, num_decodes, B, 1]
    rew_pred = decode_outputs.rew_pred
    rew_recon_loss = optax.squared_error(rew_pred, dec_rewards)
    rew_recon_loss = rew_recon_loss.sum(axis=0).sum(axis=0).mean()

    # kl loss
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
    posterior = tfd.Normal(loc=mu, scale=jnp.exp(logS))
    prior = tfd.Normal(loc=m, scale=jnp.exp(logE))
    kld = tfd.kl_divergence(posterior, prior).sum(axis=-1)
    kld = kld.sum(axis=0).sum(axis=0).mean()

    total_loss = (
        config.vae.kl_weight * kld + config.vae.rew_recon_weight * rew_recon_loss
    )

    loss_dict = {
        "kld": kld,
        "rew_recon_loss": rew_recon_loss,
    }

    return total_loss, loss_dict


def update_vae(
    ts: TrainState,
    config: FrozenConfigDict,
    batch,
    rng_key: PRNGKey,
    decode_fn: Callable,
):
    prev_obs, next_obs, actions, rewards, tasks, trajectory_lens = batch
    batch = Batch(
        prev_obs=prev_obs,
        next_obs=next_obs,
        actions=actions,
        rewards=rewards,
    )
    ts, (total_loss, loss_dict) = update_jit(
        ts=ts,
        batch=batch,
        config=config,
        rng_key=rng_key,
        decode_fn=decode_fn,
    )
    return ts, (total_loss, loss_dict)


@partial(jax.jit, static_argnames=("config", "decode_fn"))
def update_jit(
    ts: TrainState,
    config: FrozenConfigDict,
    rng_key: PRNGKey,
    batch: Batch,
    decode_fn: Callable,
):
    logging.info("update jit vae!!!!! VAEE")
    loss_and_grad_fn = jax.value_and_grad(loss_vae, has_aux=True)

    (vae_loss, metrics), grads = loss_and_grad_fn(
        ts.params,
        ts=ts,
        config=config,
        rng_key=rng_key,
        batch=batch,
        decode_fn=decode_fn,
    )

    # compute norm of grads for each set of parameters
    _, stats = gutl.compute_all_grad_norm(grad_norm_type="2", grads=grads)
    metrics.update(stats)
    ts = ts.apply_gradients(grads=grads)
    return ts, (vae_loss, metrics)
