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
from varibad_jax.models.helpers import encode_trajectory, decode
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

        continuous_actions = not isinstance(self.envs.action_space, gym.spaces.Discrete)
        if continuous_actions:
            self.vae_action_dim = self.action_dim
        else:
            self.vae_action_dim = 1
        self.continuous_actions = continuous_actions

        # this is for the case with fixed length sessions
        steps_per_rollout = (
            config.env.num_episodes_per_rollout * self.envs.max_episode_steps
        )
        self.steps_per_rollout = steps_per_rollout

        self.num_updates = (
            int(config.env.num_frames) // steps_per_rollout // config.env.num_processes
        )
        logging.info(f"num rl updates = {self.num_updates}")
        logging.info(f"steps per rollout = {steps_per_rollout}")
        logging.info(
            f"vae action dim = {self.vae_action_dim}, action_dim = {self.action_dim}"
        )

        self.ts_vae, self.ts_policy = self.create_ts()
        self.policy_storage, self.vae_storage = self.setup_replay_buffers()

    def create_ts(self):
        vae_params = init_params_vae(
            config=self.config.vae,
            rng_key=next(self.rng_seq),
            state_dim=self.state_dim,
            action_dim=self.vae_action_dim,
        )

        # count number of vae params
        num_vae_params = sum(p.size for p in jax.tree_util.tree_leaves(vae_params))
        logging.info(f"num vae params = {num_vae_params}")

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

        num_policy_params = sum(
            p.size for p in jax.tree_util.tree_leaves(policy_params)
        )
        logging.info(f"num policy params = {num_policy_params}")

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

        self.decode_apply = partial(
            jax.jit(
                decode.apply,
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
            is_continuous=self.continuous_actions,
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
            action_dim=self.vae_action_dim,
            task_dim=0,
            vae_buffer_add_thresh=1.0,
        )
        return policy_storage, vae_storage

    def collect_rollouts(self):
        logging.debug("inside rollout")

        # use all zeros as priors
        latent = np.zeros((self.num_processes, self.config.vae.latent_dim * 2))
        latent_sample = np.zeros((self.num_processes, self.config.vae.latent_dim))
        latent_mean = np.zeros((self.num_processes, self.config.vae.latent_dim))
        latent_logvar = np.zeros((self.num_processes, self.config.vae.latent_dim))
        hidden_state = np.zeros((self.num_processes, self.config.vae.lstm_hidden_size))

        state = self.envs.reset()
        if len(state.shape) == 1:  # add extra dimension
            state = state[..., np.newaxis]

        self.policy_storage.prev_state[0] = state
        self.policy_storage.hidden_states[0] = hidden_state.copy()
        self.policy_storage.latent_samples.append(latent_sample.copy())
        self.policy_storage.latent_mean.append(latent_mean.copy())
        self.policy_storage.latent_logvar.append(latent_logvar.copy())

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

        logging.debug("done collecting rollouts")

    def perform_update(self):
        logging.debug("inside perform update")
        metrics = {}

        # compute ppo update
        update_start = time.time()
        policy_metrics, self.ts_policy = update_policy(
            ts=self.ts_policy,
            replay_buffer=self.policy_storage,
            config=FrozenConfigDict(self.config.policy),
            rng_key=next(self.rng_seq),
        )
        policy_update_time = time.time() - update_start
        policy_metrics = gutl.prefix_dict_keys(policy_metrics, prefix="policy/")
        metrics.update(policy_metrics)
        logging.debug("done updating policy")

        # update vae
        update_start = time.time()
        for _ in range(self.config.vae.num_vae_updates):
            self.ts_vae, (_, vae_metrics) = update_vae(
                ts=self.ts_vae,
                rng_key=next(self.rng_seq),
                config=FrozenConfigDict(self.config),
                batch=self.vae_storage.get_batch(self.config.vae.trajs_per_batch),
                decode_fn=self.decode_apply,
            )
        vae_update_time = time.time() - update_start
        vae_metrics = gutl.prefix_dict_keys(vae_metrics, prefix="vae/")
        metrics.update(vae_metrics)
        metrics["time/vae_update_time"] = vae_update_time
        metrics["time/policy_update_time"] = policy_update_time

        logging.debug("done updating vae")
        return metrics

    def train(self):
        logging.info("Training starts")

        for self.iter_idx in tqdm.tqdm(
            range(0, self.num_updates), smoothing=0.1, disable=self.config.disable_tqdm
        ):
            print(
                "------------------ Iteration {}; Frames {} ------------------".format(
                    self.iter_idx, self.total_steps
                )
            )
            # collect rollout
            rollout_start = time.time()
            self.collect_rollouts()
            rollout_time = time.time() - rollout_start

            # Log episodic rewards for completed environments
            # sum over timesteps and average over environments
            episodic_rewards_raw = self.policy_storage.rewards_raw.sum(axis=0).mean()
            episodic_rewards_normalised = self.policy_storage.rewards_normalised.sum(
                axis=0
            ).mean()
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "train/episodic_rewards_raw": episodic_rewards_raw,
                        "train/episodic_rewards_normalised": episodic_rewards_normalised,
                    },
                    step=self.iter_idx,
                )

            if (
                len(self.vae_storage) == 0 and not self.config.vae.buffer_size == 0
            ) or (self.config.warmup_steps > self.total_steps):
                logging.info("Not updating yet because; filling up the VAE buffer.")
                self.policy_storage.after_update()
                continue

            # update vae and policy
            # wait for some warmup steps first
            if self.total_steps >= self.config.warmup_steps:
                train_metrics = self.perform_update()

                if ((self.iter_idx + 1) % self.config.log_interval) == 0:
                    # Environment stats
                    env_stats = {
                        "state_max": self.policy_storage.prev_state.max(),
                        "state_min": self.policy_storage.prev_state.min(),
                        "rew_max": self.policy_storage.rewards_raw.max(),
                        "rew_min": self.policy_storage.rewards_raw.min(),
                    }
                    env_stats = gutl.prefix_dict_keys(env_stats, prefix="env/")
                    metrics = {
                        **train_metrics,
                        **env_stats,
                        "time/rollout_time": rollout_time,
                    }
                    if self.wandb_run is not None:
                        self.wandb_run.log(metrics, step=self.iter_idx)

                # if (self.iter_idx + 1) % self.config.eval_interval == 0:
                #   print('Evaluating...')
                #   returns_per_episode = evaluate(
                #       config=self.config,
                #       ts_policy=self.ts_policy,
                #       ts_vae=self.ts_vae,
                #       rng_seq=self.rng_seq,
                #       transition_matrix=self.goal_transition_matrix,
                #       video_dir=self.eval_video_dir,
                #       iter_idx=self.iter_idx + 1,
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
                #       self.iter_idx=self.iter_idx,
                #   )

                if ((self.iter_idx + 1) % self.config.save_interval) == 0:
                    # # Bundle everything together.
                    # self.saver.save(
                    #     iter_idx=self.iter_idx + 1,
                    #     ckpt={
                    #         "config": self.config.to_dict(),
                    #         "ts_policy": self.ts_policy,
                    #         "ts_vae": self.ts_vae,
                    #     },
                    # )
                    # save to pickle
                    ckpt_file = Path(self.ckpt_dir) / f"ckpt_{self.iter_idx}.pkl"
                    logging.debug(f"saving checkpoint to {ckpt_file}")
                    with open(ckpt_file, "wb") as f:
                        pickle.dump(
                            {
                                "config": self.config.to_dict(),
                                "ts_policy": self.ts_policy.params,
                                "ts_vae": self.ts_vae.params,
                            },
                            f,
                        )

            # Clean up after update
            self.policy_storage.after_update()

        self.envs.close()
        logging.info("Finished training, closing everything")
        return
