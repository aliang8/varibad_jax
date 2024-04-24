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

from varibad_jax.models.varibad.helpers import Batch, encode_trajectory, decode
from varibad_jax.models.hyperx.helpers import compute_hyperx_bonus

tfd = tfp.distributions
tfb = tfp.bijectors

@partial(jax.jit, static_argnames=("config", "get_prior_fn"))
def compute_hyperx_loss(
    predictor_params,
    ts_predictor: TrainState,
    ts_prior: TrainState,
    ts_vae: TrainState,
    state: hk.State,
    vae_state: hk.State,
    config: FrozenConfigDict,
    batch: Batch,
    rng_key: PRNGKey,
    get_prior_fn: Callable
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

    hyperstate_bonus, state = compute_hyperx_bonus(predictor_params, ts_prior, ts_predictor, rng=hyperx_key, obs=batch.next_obs, latent=latent_belief)

    loss = jnp.mean(hyperstate_bonus)
    return loss, state

def update_hyperx(
    ts_predictor: TrainState,
    ts_prior: TrainState,
    ts_vae: TrainState,
    state: hk.State,
    vae_state: hk.State,
    config: FrozenConfigDict,
    batch: Batch,
    rng_key: PRNGKey,
    get_prior_fn: Callable,
):
    prev_obs, next_obs, actions, rewards, tasks, trajectory_lens = batch
    batch = Batch(
        prev_obs=prev_obs,
        next_obs=next_obs,
        actions=actions,
        rewards=rewards,
    )

    loss_and_grad_fn = jax.value_and_grad(compute_hyperx_loss, has_aux=True)

    (total_loss, state), grads = loss_and_grad_fn(
        ts_predictor.params,
        ts_predictor=ts_predictor,
        ts_prior=ts_prior,
        ts_vae=ts_vae,
        state=state,
        vae_state=vae_state,
        batch=batch,
        config=config,
        rng_key=rng_key,
        get_prior_fn=get_prior_fn,
    )
    # update the predictor model and not the prior model
    ts_predictor = ts_predictor.apply_gradients(grads=grads)
    return ts_predictor, (total_loss, state)