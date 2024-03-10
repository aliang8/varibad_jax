import os
import collections
import random
import time
import sys
from absl import app, flags
from absl import logging
import pickle
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import numpy as np
import optax
import tqdm
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
from functools import partial
import haiku as hk
from pathlib import Path
import gymnasium as gym
from varibad_jax.base_trainer import BaseTrainer
from varibad_jax.models.helpers import init_params as init_params_vae
from varibad_jax.models.helpers import encode_trajectory
from varibad_jax.models.update import update_vae
from varibad_jax.agents.ppo.ppo import update_policy
from varibad_jax.agents.ppo.helpers import init_params as init_params_policy
from varibad_jax.agents.ppo.helpers import policy_fn

from varibad_jax.utils.replay_buffer import OnlineStorage
from varibad_jax.utils.replay_vae import RolloutStorageVAE
import varibad_jax.utils.general_utils as gutl


class VAETrainer(BaseTrainer):
    def __init__(self, config: ConfigDict):
        super().__init__(config)
        self.total_steps = 0
        self.num_processes = self.config.env.num_processes

        # this is for the case with fixed length sessions
        steps_per_rollout = (
            config.env.num_episodes_per_rollout * self.envs.max_episode_steps
        )
        self.steps_per_rollout = steps_per_rollout

        self.num_updates = (
            int(config.env.num_frames) // steps_per_rollout // config.env.num_processes
        )
        print(f"num rl updates = {self.num_updates}")

        self.ts_vae, self.ts_policy = self.create_ts()
        self.policy_storage, self.vae_storage = self.setup_replay_buffers()

    def create_ts(self):
        continuous_actions = not isinstance(self.envs.action_space, gym.spaces.Discrete)
        if continuous_actions:
            action_dim = self.action_dim
        else:
            action_dim = 1
        vae_params = init_params_vae(
            config=self.config.vae,
            rng_key=next(self.rng_seq),
            state_dim=self.state_dim,
            action_dim=action_dim,
        )

        tx_vae = optax.chain(
            optax.clip(self.config.vae.max_grad_norm),
            optax.adam(self.config.vae.lr, eps=self.config.vae.eps),
        )

        policy_params = init_params_policy(
            config=self.config.policy,
            rng_key=next(self.rng_seq),
            state_dim=self.state_dim,
            latent_dim=self.config.vae.latent_dim * 2,
            action_space=self.envs.action_space,
        )

        tx_policy = optax.chain(
            optax.clip(self.config.policy.max_grad_norm),
            optax.adam(self.config.policy.lr, eps=self.config.policy.eps),
        )

        encode_apply = partial(
            jax.jit(
                encode_trajectory.apply,
                static_argnames=("config"),
            ),
            config=FrozenConfigDict(self.config.vae),
        )

        ts_vae = TrainState.create(apply_fn=encode_apply, params=vae_params, tx=tx_vae)

        policy_apply = partial(
            jax.jit(
                policy_fn.apply,
                static_argnames=("config", "is_continuous", "action_dim"),
            ),
            config=FrozenConfigDict(self.config.policy),
            is_continuous=continuous_actions,
            action_dim=self.action_dim,
        )
        ts_policy = TrainState.create(
            apply_fn=policy_apply,
            params=policy_params,
            tx=tx_policy,
        )
        return ts_vae, ts_policy

    def setup_replay_buffers(self):
        policy_storage = OnlineStorage(
            args=self.config,
            num_steps=self.steps_per_rollout,
            num_processes=self.num_processes,
            state_dim=self.state_dim,
            latent_dim=self.config.vae.latent_dim * 2,
            belief_dim=0,
            task_dim=0,
            action_space=self.envs.action_space,
            hidden_size=self.config.vae.lstm_hidden_size,
            normalise_rewards=self.config.env.normalize_rews,
        )

        vae_storage = RolloutStorageVAE(
            num_processes=self.num_processes,
            max_trajectory_len=self.steps_per_rollout,
            zero_pad=True,
            max_num_rollouts=self.config.vae.buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            task_dim=0,
            vae_buffer_add_thresh=1.0,
        )
        return policy_storage, vae_storage

    def collect_rollouts(self):
        # use all zeros as priors
        latent = np.zeros((self.num_processes, self.config.vae.latent_dim * 2))
        hidden_state = np.zeros((self.num_processes, self.config.vae.lstm_hidden_size))

        state = self.envs.reset()
        if len(state.shape) == 1:  # add extra dimension
            state = state[..., np.newaxis]

        self.policy_storage.prev_state[0] = state
        self.policy_storage.hidden_states[0] = hidden_state.copy()

        for step in range(self.steps_per_rollout):
            # sample random action from policy
            policy_output = self.ts_policy.apply_fn(
                self.ts_policy.params,
                next(self.rng_seq),
                env_state=state,
                latent=latent,
            )

            # take a step in the environment
            action = policy_output.action
            if len(action.shape) == 1:
                action = action[..., np.newaxis]

            next_state, rew_raw, done, infos = self.envs.step(action)
            rew_norm = rew_raw

            # add extra dimension
            if len(next_state.shape) == 1:
                next_state = next_state[np.newaxis]

            done = done[:, np.newaxis]
            reward = rew_raw[:, np.newaxis]
            reward_norm = rew_norm[:, np.newaxis]
            value = policy_output.value
            log_prob = policy_output.log_prob[:, np.newaxis]

            # after seeing the next observation and getting more
            # information, we can update the belief over the task variable
            encode_outputs = self.ts_vae.apply_fn(
                self.ts_vae.params,
                next(self.rng_seq),
                states=next_state,
                actions=action,
                rewards=reward,
                hidden_state=hidden_state,
            )

            # add transition to vae buffer
            self.vae_storage.insert(
                prev_state=state,
                next_state=next_state,
                actions=action,
                rewards=reward,
                done=done,
                task=None,
            )

            self.policy_storage.next_state[step] = next_state.copy()

            # mask out timesteps that are done
            masks_done = np.array([[0.0] if done_ else [1.0] for done_ in done])

            # add experience to policy buffer
            self.policy_storage.insert(
                state=next_state,
                belief=None,
                task=None,
                actions=action,
                rewards_raw=reward,
                rewards_normalised=reward_norm,
                masks=masks_done,
                bad_masks=masks_done,  # TODO: fix this?
                value_preds=value,
                done=done,
                hidden_states=hidden_state,
                latent_mean=encode_outputs.latent_mean,
                latent_logvar=encode_outputs.latent_logvar,
                latent_sample=encode_outputs.latent_mean,
            )

            # update state
            state = next_state
            latent = encode_outputs.latent
            self.total_steps += self.num_processes

        # compute next value for bootstrapping
        policy_output = self.ts_policy.apply_fn(
            self.ts_policy.params,
            next(self.rng_seq),
            env_state=state,
            latent=latent,
        )
        next_value = policy_output.value

        # compute returns for current rollout
        self.policy_storage.compute_returns(
            next_value=next_value,
            use_gae=True,
            gamma=self.config.policy.gamma,
            tau=self.config.policy.tau,
            use_proper_time_limits=False,
        )

    def perform_update(self):
        metrics = {}

        # compute ppo update
        update_start = time.time()
        policy_metrics, self.ts_policy = update_policy(
            ts=self.ts_policy,
            replay_buffer=self.policy_storage,
            config=self.config.policy,
            rng_key=next(self.rng_seq),
        )
        update_time = time.time() - update_start
        policy_metrics["update_time"] = update_time
        policy_metrics = gutl.prefix_dict_keys(policy_metrics, prefix="policy/")
        metrics.update(policy_metrics)

        # update vae
        update_start = time.time()
        for _ in range(self.config.vae.num_vae_updates):
            self.ts_vae, (_, vae_metrics) = update_vae(
                ts=self.ts_vae,
                rng_key=next(self.rng_seq),
                config=self.config,
                batch=self.vae_storage.get_batch(self.config.vae.trajs_per_batch),
            )
        update_time = time.time() - update_start
        vae_metrics["update_time"] = update_time
        vae_metrics = gutl.prefix_dict_keys(vae_metrics, prefix="vae/")
        metrics.update(vae_metrics)
        return metrics

    def train(self):
        print("Training starts")

        for iter_idx in tqdm.tqdm(
            range(0, self.num_updates), smoothing=0.1, disable=self.config.disable_tqdm
        ):
            print(
                "------------------ Iteration {}; Frames {} ------------------".format(
                    iter_idx, self.total_steps
                )
            )
            # collect rollout
            self.collect_rollouts()

            if (
                len(self.vae_storage) == 0
                and not self.config.vae.buffer_size == 0
            ) or (self.config.warmup_steps > self.total_steps):
                print("Not updating yet because; filling up the VAE buffer.")
                self.policy_storage.after_update()
                continue

            # update vae and policy
            # wait for some warmup steps first
            if self.total_steps >= self.config.warmup_steps:
                train_metrics = self.perform_update()

                if ((iter_idx + 1) % self.config.log_interval) == 0:
                    # Environment stats
                    env_stats = {
                        "state_max": self.policy_storage.prev_state.max(),
                        "state_min": self.policy_storage.prev_state.min(),
                        "rew_max": self.policy_storage.rewards_raw.max(),
                        "rew_min": self.policy_storage.rewards_raw.min(),
                    }
                    train_metrics.update(env_stats)
                    self.log(train_metrics, iter_idx=iter_idx)

                # if (iter_idx + 1) % self.config.eval_interval == 0:
                #   print('Evaluating...')
                #   returns_per_episode = evaluate(
                #       config=self.config,
                #       ts_policy=self.ts_policy,
                #       ts_vae=self.ts_vae,
                #       rng_seq=self.rng_seq,
                #       transition_matrix=self.goal_transition_matrix,
                #       video_dir=self.eval_video_dir,
                #       iter_idx=iter_idx + 1,
                #   )
                #   avg_returns_per_episode = returns_per_episode.mean(axis=0)

                #   eval_metrics = {
                #       f'returns_per_episode/avg/episode_{j}': avg_returns_per_episode[j]
                #       for j in range(len(avg_returns_per_episode))
                #   }
                #   eval_metrics['returns_per_episode/avg'] = returns_per_episode.mean()
                #   self.log(
                #       eval_metrics,
                #       phase='eval',
                #       iter_idx=iter_idx,
                #   )

                if ((iter_idx + 1) % self.config.save_interval) == 0:
                    # Bundle everything together.
                    self.saver.save(
                        iter_idx=iter_idx + 1,
                        ckpt={
                            "config": self.config.to_dict(),
                            "ts_policy": self.ts_policy,
                            "ts_vae": self.ts_vae,
                        },
                    )

            # Clean up after update
            self.policy_storage.after_update()

        self.envs.close()
        print("Finished training, closing everything")
        return
