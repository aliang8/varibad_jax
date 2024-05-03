from absl import logging
import jax
import chex
import einops
import optax
import numpy as np
import haiku as hk
from typing import Any
import jax.numpy as jnp
import flax.linen as nn
from jax.random import PRNGKey
from ml_collections.config_dict import ConfigDict
import gymnasium as gym
from functools import partial
from collections import defaultdict as dd
from tensorflow_probability.substrates import jax as tfp

from varibad_jax.agents.actor_critic import ActorCritic
from varibad_jax.models.base import BaseAgent

tfd = tfp.distributions
tfb = tfp.bijectors


class PPOAgent(BaseAgent):
    @hk.transform_with_state
    def model(self, env_state, **kwargs):
        policy = ActorCritic(self.config, self.is_continuous, self.action_dim)
        return policy(state=env_state, **kwargs)

    def _init_model(self):
        t, bs = 2, 2
        dummy_states = np.zeros((t, bs, *self.observation_shape), dtype=np.float32)
        if self.config.pass_latent_to_policy:
            dummy_latents = np.zeros((t, bs, self.config.latent_dim * 2))
        else:
            dummy_latents = None

        if self.config.pass_task_to_policy:
            dummy_tasks = np.zeros((t, bs, self.config.task_dim))
        else:
            dummy_tasks = None

        self._params, self._state = self.model.init(
            self._init_key,
            self,
            env_state=dummy_states,
            latent=dummy_latents,
            task=dummy_tasks,
            is_training=True,
        )

    def loss_fn(
        self,
        params: hk.Params,
        state: hk.State,
        rng_key: PRNGKey,
        env_state: jnp.ndarray,
        target: jnp.ndarray,
        value_old: jnp.ndarray,
        log_pi_old: jnp.ndarray,
        gae: jnp.ndarray,
        action: jnp.ndarray,
        latent: jnp.ndarray = None,
        task: jnp.ndarray = None,
    ):
        policy_output, state = self.model.apply(
            params,
            state,
            rng_key,
            self,
            env_state=env_state,
            latent=latent,
            task=task,
            is_training=True,
        )
        policy_dist = policy_output.dist
        value_pred = policy_output.value

        log_prob = policy_dist.log_prob(action.astype(np.int32).squeeze())
        if len(log_prob.shape) == 1:
            log_prob = jnp.expand_dims(log_prob, axis=-1)

        value_pred_clipped = value_old + (value_pred - value_old).clip(
            -self.config.clip_eps, self.config.clip_eps
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
            jnp.clip(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
            * gae_norm
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()

        entropy = policy_dist.entropy().mean()

        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-log_ratio).mean()
        approx_kl = ((ratio - 1) - log_ratio).mean()
        # clipfracs = [((ratio - 1.0).abs() > self.config.clip_eps).float().mean()]

        total_loss = (
            loss_actor
            + self.config.value_loss_coeff * value_loss
            - self.config.entropy_coeff * entropy
        )
        metrics = {
            "value_loss": value_loss,
            "actor_loss": loss_actor,
            "entropy": entropy,
            "value_pred_mean": value_pred.mean(),
            "target_mean": target.mean(),
            "gae_mean": gae.mean(),
            "approx_kl": approx_kl,
        }
        return total_loss, (metrics, state)

    @partial(jax.jit, static_argnums=(0,))
    def update_model(
        self,
        params: hk.Params,
        state: hk.State,
        opt_state,
        rng_key: PRNGKey,
        idxes: np.ndarray,
        env_state: np.ndarray,
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
            loss_and_grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
            (_, (loss_dict, state)), grads = loss_and_grad_fn(
                params,
                state,
                rng_key,
                env_state=env_state[idx],
                target=target[idx],
                value_old=value[idx],
                log_pi_old=log_pi_old[idx],
                gae=gae[idx],
                action=action[idx],
                latent=latent[idx] if latent is not None else latent,
                task=task[idx] if task is not None else task,
            )
            # grad_norms, stats = gutl.compute_all_grad_norm(grad_norm_type="2", grads=grads)
            # loss_dict.update(stats)
            grads, opt_state = self.opt.update(grads, opt_state)
            params = optax.apply_updates(params, grads)

        return params, state, opt_state, loss_dict

    def update(self, rng: jax.random.PRNGKey, replay_buffer):
        _, key1, key2 = jax.random.split(rng, 3)
        self._state = replay_buffer.before_update(self, key1)
        num_steps, num_processes = replay_buffer.rewards_raw.shape[:2]
        size_batch = num_processes * num_steps
        size_minibatch = size_batch // self.config.num_minibatch
        idxes = np.arange(size_batch)

        metrics = dd(int)

        # flatten T and B dimension
        flatten = lambda x: einops.rearrange(x, "T B ... -> (T B) ...")
        env_state = flatten(replay_buffer.prev_state[:-1])
        action = flatten(replay_buffer.actions)
        value = flatten(replay_buffer.value_preds[:-1])
        log_pi_old = flatten(replay_buffer.action_log_probs)
        target = flatten(replay_buffer.returns[:-1])
        advantages = target - value
        gae = advantages

        if self.config.pass_latent_to_policy:
            latent_mean = np.array(replay_buffer.latent_mean[:-1])
            latent_logvar = np.array(replay_buffer.latent_logvar[:-1])
            latent = np.concatenate([latent_mean, latent_logvar], axis=-1)
            latent = flatten(latent)
        else:
            latent = None

        if self.config.pass_task_to_policy:
            task = flatten(np.array(replay_buffer.tasks[:-1]))
        else:
            task = None

        for _ in range(self.config.num_epochs):
            idxes = np.random.permutation(idxes)
            idxes_list = [
                idxes[start : start + size_minibatch]
                for start in jnp.arange(0, size_batch, size_minibatch)
            ]
            self._params, self._state, self._opt_state, epoch_metrics = (
                self.update_model(
                    params=self._params,
                    state=self._state,
                    opt_state=self._opt_state,
                    rng_key=key2,
                    idxes=idxes_list,
                    env_state=env_state,
                    action=action,
                    log_pi_old=log_pi_old,
                    value=value,
                    target=target,
                    gae=gae,
                    latent=latent,
                    task=task,
                )
            )

            for k, v in epoch_metrics.items():
                metrics[k] += np.asarray(v)

        for k, v in metrics.items():
            metrics[k] = v / (self.config.num_epochs)

        return metrics
