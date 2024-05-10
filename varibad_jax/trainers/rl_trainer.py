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

from varibad_jax.trainers.base_trainer import BaseTrainer
from varibad_jax.agents.ppo.ppo import PPOAgent
from varibad_jax.utils.rollout import run_rollouts

from varibad_jax.utils.replay_buffer import OnlineStorage
import varibad_jax.utils.general_utils as gutl


class RLTrainer(BaseTrainer):
    def __init__(self, config: ConfigDict):
        super().__init__(config)
        self.total_steps = 0
        self.num_processes = config.env.num_processes
        self.num_updates = (
            int(config.num_frames) // self.steps_per_rollout // config.env.num_processes
        )
        logging.info(f"num rl updates = {self.num_updates}")
        logging.info(f"steps per rollout = {self.steps_per_rollout}")
        logging.info(f"action_dim = {self.action_dim}")

        self.agent = PPOAgent(
            config=config.model,
            observation_shape=self.envs.observation_space.shape,
            action_dim=self.action_dim,
            input_action_dim=self.input_action_dim,
            continuous_actions=self.continuous_actions,
            key=next(self.rng_seq),
        )
        self.policy_storage = self.setup_replay_buffers()

    def setup_replay_buffers(self):
        policy_storage = OnlineStorage(
            args=self.config,
            num_steps=self.steps_per_rollout,
            num_processes=self.num_processes,
            state_dim=self.envs.observation_space.shape,
            latent_dim=0,
            belief_dim=0,
            task_dim=self.task_dim,
            action_space=self.envs.action_space,
            hidden_size=0,
            normalise_rewards=self.config.env.normalize_rews,
        )
        return policy_storage

    def collect_rollouts(self):
        logging.debug("inside rollout")
        reset_rng = next(self.rng_seq)
        reset_rng = jax.random.split(reset_rng, self.num_processes)
        xtimestep = self.jit_reset(self.env_params, reset_rng)
        state = xtimestep.timestep.observation
        state = state.astype(np.float32)
        if self.config.model.policy.pass_task_to_policy:
            if self.config.env.env_name == "gridworld":
                task = xtimestep.timestep.state.goal
            elif self.config.env.env_name == "xland":
                task = xtimestep.timestep.state.goal_encoding

            if len(task.shape) == 1:
                task = task[np.newaxis]
        else:
            task = None

        if len(state.shape) == 1:  # add extra dimension
            state = state[..., np.newaxis]

        hidden_state = None

        self.policy_storage.prev_state[0] = state

        for step in range(self.steps_per_rollout):
            # sample random action from policy
            (policy_output, hidden_state), self.agent._state = self.agent.get_action(
                next(self.rng_seq),
                env_state=state,
                task=task,
                hidden_state=hidden_state,
                is_training=True,
            )

            # take a step in the environment
            action = policy_output.action
            xtimestep = self.jit_step(self.env_params, xtimestep, action)
            next_state = xtimestep.timestep.observation
            next_state = next_state.astype(np.float32)
            rew_raw = xtimestep.timestep.reward
            done = xtimestep.timestep.last()
            rew_norm = rew_raw

            if self.config.model.policy.pass_task_to_policy:
                if self.config.env.env_name == "gridworld":
                    task = xtimestep.timestep.state.goal
                elif self.config.env.env_name == "xland":
                    task = xtimestep.timestep.state.goal_encoding

                if len(task.shape) == 1:
                    task = task[np.newaxis]
            else:
                task = None

            # add extra dimension
            if len(next_state.shape) == 1:
                next_state = next_state[np.newaxis]

            if len(action.shape) == 1:
                action = action[:, np.newaxis]

            done = done[:, np.newaxis]
            reward = rew_raw[:, np.newaxis]
            reward_norm = rew_norm[:, np.newaxis]
            value = policy_output.value

            self.policy_storage.next_state[step] = next_state.copy()

            # mask out timesteps that are done
            masks_done = np.array([[0.0] if done_ else [1.0] for done_ in done])

            # add experience to policy buffer
            self.policy_storage.insert(
                state=next_state,
                belief=None,
                task=task,
                actions=action,
                rewards_raw=reward,
                rewards_normalised=reward_norm,
                hyperx_bonuses=None,
                vae_recon_bonuses=None,
                masks=masks_done,
                bad_masks=masks_done,  # TODO: fix this?
                value_preds=value,
                done=done,
            )

            # update state
            state = next_state
            self.total_steps += self.num_processes

        # compute next value for bootstrapping
        (policy_output, _), self.agent._state = self.agent.get_action(
            next(self.rng_seq),
            env_state=state,
            task=task,
            hidden_state=hidden_state,
            is_training=True,
        )
        next_value = policy_output.value

        # compute returns for current rollout
        self.policy_storage.compute_returns(
            next_value=next_value,
            use_gae=True,
            gamma=self.config.model.gamma,
            tau=self.config.model.tau,
            use_proper_time_limits=False,
        )
        logging.debug("done collecting rollouts")

    def perform_update(self):
        logging.debug("inside perform update")
        metrics = {}

        # compute ppo update
        update_start = time.time()
        policy_metrics = self.agent.update(
            replay_buffer=self.policy_storage,
            rng=next(self.rng_seq),
        )
        policy_update_time = time.time() - update_start
        policy_metrics = gutl.prefix_dict_keys(policy_metrics, prefix="policy/")
        metrics.update(policy_metrics)
        logging.debug("done updating policy")
        metrics["time/policy_update_time"] = policy_update_time
        return metrics

    def train(self):
        logging.info("Training starts")
        start_time = time.time()

        # first eval
        if not self.config.skip_first_eval:
            eval_metrics, *_ = run_rollouts(
                rng=next(self.rng_seq),
                agent=self.agent,
                env=self.eval_envs,
                config=self.config,
                steps_per_rollout=self.steps_per_rollout,
                action_dim=self.input_action_dim,
                wandb_run=self.wandb_run,
            )

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
            # import ipdb

            # ipdb.set_trace()
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

            if self.config.warmup_steps > self.total_steps:
                logging.info("Not updating yet ...")
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
                        "rew_avg": self.policy_storage.rewards_raw.mean(),
                        "rew_sum": self.policy_storage.rewards_raw.sum(),
                        "rew_goal": (self.policy_storage.rewards_raw == 1).sum(),
                    }
                    env_stats = gutl.prefix_dict_keys(env_stats, prefix="env/")
                    metrics = {
                        **train_metrics,
                        **env_stats,
                        "time/rollout_time": rollout_time,
                        "time/fps": self.total_steps / (time.time() - start_time),
                    }
                    if self.wandb_run is not None:
                        self.wandb_run.log(metrics, step=self.iter_idx)

                if (self.iter_idx + 1) % self.config.eval_interval == 0:
                    eval_metrics, *_ = run_rollouts(
                        rng=next(self.rng_seq),
                        agent=self.agent,
                        env=self.eval_envs,
                        config=self.config,
                        steps_per_rollout=self.steps_per_rollout,
                        action_dim=self.input_action_dim,
                        wandb_run=self.wandb_run,
                    )
                    # save to pickle
                    self.save_model(
                        self.agent.save_dict, eval_metrics, iter=self.iter_idx
                    )

                    if self.wandb_run is not None:
                        eval_metrics = gutl.prefix_dict_keys(eval_metrics, "eval/")
                        self.wandb_run.log(eval_metrics, step=self.iter_idx)

            # Clean up after update
            self.policy_storage.after_update()

        self.envs.close()
        logging.info("Finished training, closing everything")
        return
