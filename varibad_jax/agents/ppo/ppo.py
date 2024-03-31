from collections import defaultdict
from functools import partial
from typing import Any, Callable, Tuple

from absl import logging
import flax
from flax import struct
import flax.linen as nn
from flax.training.train_state import TrainState
import haiku as hk
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from ml_collections.config_dict import ConfigDict
import numpy as np
from tensorflow_probability.substrates import jax as tfp
import varibad_jax.utils.general_utils as gutl
from varibad_jax.agents.ppo.helpers import policy_fn

tfd = tfp.distributions
tfb = tfp.bijectors


def loss_policy(
    params,
    rng_key: PRNGKey,
    ts: TrainState,
    config: ConfigDict,
    state: jnp.ndarray,
    target: jnp.ndarray,
    value_old: jnp.ndarray,
    log_pi_old: jnp.ndarray,
    gae: jnp.ndarray,
    action: jnp.ndarray,
    latent: jnp.ndarray = None,
    task: jnp.ndarray = None,
):
    policy_output = ts.apply_fn(
        params, rng_key, env_state=state, latent=latent, task=task
    )
    policy_dist = policy_output.dist
    value_pred = policy_output.value

    log_prob = policy_dist.log_prob(action.astype(np.int32).squeeze())
    if len(log_prob.shape) == 1:
        log_prob = jnp.expand_dims(log_prob, axis=-1)

    value_pred_clipped = value_old + (value_pred - value_old).clip(
        -config.clip_eps, config.clip_eps
    )
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    log_ratio = log_prob - log_pi_old
    ratio = jnp.exp(log_ratio)

    # import ipdb

    # ipdb.set_trace()

    gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae_norm
    loss_actor2 = (
        jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * gae_norm
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    entropy = policy_dist.entropy().mean()

    # calculate approx_kl http://joschu.net/blog/kl-approx.html
    old_approx_kl = (-log_ratio).mean()
    approx_kl = ((ratio - 1) - log_ratio).mean()
    # clipfracs = [((ratio - 1.0).abs() > config.clip_eps).float().mean()]

    total_loss = (
        loss_actor
        + config.value_loss_coeff * value_loss
        - config.entropy_coeff * entropy
    )

    return total_loss, {
        "value_loss": value_loss,
        "actor_loss": loss_actor,
        "entropy": entropy,
        "value_pred_mean": value_pred.mean(),
        "target_mean": target.mean(),
        "gae_mean": gae.mean(),
        "approx_kl": approx_kl,
    }


@partial(jax.jit, static_argnames="config")
def update_jit(
    ts: TrainState,
    config: ConfigDict,
    rng_key: PRNGKey,
    idxes: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    log_pi_old: np.ndarray,
    value: np.ndarray,
    target: np.ndarray,
    gae: np.ndarray,
    latent: np.ndarray = None,
    task: np.ndarray = None,
):
    logging.info("update jit policy! POLICY!!!")
    for idx in idxes:
        loss_and_grad_fn = jax.value_and_grad(loss_policy, has_aux=True)
        (_, loss_dict), grads = loss_and_grad_fn(
            ts.params,
            rng_key,
            ts=ts,
            config=config,
            state=state[idx],
            target=target[idx],
            value_old=value[idx],
            log_pi_old=log_pi_old[idx],
            gae=gae[idx],
            action=action[idx],
            latent=latent[idx] if latent is not None else latent,
            task=task[idx] if task is not None else task,
        )
        grad_norms, stats = gutl.compute_all_grad_norm(grad_norm_type="2", grads=grads)
        loss_dict.update(stats)
        ts = ts.apply_gradients(grads=grads)

    return ts, (_, loss_dict)


def update_policy(
    ts: TrainState,
    replay_buffer,
    config: ConfigDict,
    rng_key: PRNGKey,
):
    _, key1, key2 = jax.random.split(rng_key, 3)
    replay_buffer.before_update(ts, key1)
    num_steps, num_processes = replay_buffer.rewards_raw.shape[:2]
    size_batch = num_processes * num_steps
    size_minibatch = size_batch // config.num_minibatch
    idxes = np.arange(size_batch)

    metrics = defaultdict(int)

    # flatten T and B dimension
    flatten = lambda x: x.reshape(-1, x.shape[-1])
    state = flatten(replay_buffer.prev_state[:-1])
    action = flatten(replay_buffer.actions)
    value = flatten(replay_buffer.value_preds[:-1])
    log_pi_old = flatten(replay_buffer.action_log_probs)
    target = flatten(replay_buffer.returns[:-1])
    advantages = target - value
    gae = advantages

    # latent = flatten(np.array(replay_buffer.latent[:-1]))
    if config.pass_latent_to_policy:
        latent_mean = np.array(replay_buffer.latent_mean[:-1])
        latent_logvar = np.array(replay_buffer.latent_logvar[:-1])
        latent = np.concatenate([latent_mean, latent_logvar], axis=-1)
        latent = flatten(latent)
    else:
        latent = None

    if config.pass_task_to_policy:
        task = flatten(np.array(replay_buffer.tasks[:-1]))
    else:
        task = None

    for _ in range(config.num_epochs):
        idxes = np.random.permutation(idxes)
        idxes_list = [
            idxes[start : start + size_minibatch]
            for start in jnp.arange(0, size_batch, size_minibatch)
        ]
        ts, (_, epoch_metrics) = update_jit(
            ts=ts,
            config=config,
            rng_key=key2,
            idxes=idxes_list,
            state=state,
            action=action,
            log_pi_old=log_pi_old,
            value=value,
            target=target,
            gae=gae,
            latent=latent,
            task=task,
        )

        for k, v in epoch_metrics.items():
            metrics[k] += np.asarray(v)

    for k, v in metrics.items():
        metrics[k] = v / (config.num_epochs)

    return metrics, ts
