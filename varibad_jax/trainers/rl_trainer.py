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

# from varibad_jax.agents.ppo.ppo import update_policy

# from varibad_jax.agents.ppo.helpers import init_params as init_params_policy
# from varibad_jax.agents.ppo.helpers import policy_fn
from varibad_jax.utils.rollout import run_rollouts

from varibad_jax.utils.replay_buffer import OnlineStorage
import varibad_jax.utils.general_utils as gutl


def create_ts(
    config: ConfigDict,
    rng,
    envs,
    continuous_actions: bool,
    action_dim: int,
    task_dim: int,
    num_update_steps: int,
):
    policy_params, policy_state = init_params_policy(
        config=config.policy,
        rng_key=rng,
        observation_space=envs.observation_space,
        latent_dim=0,
        action_space=envs.action_space,
        task_dim=task_dim,
    )
    # import ipdb

    # ipdb.set_trace()
    num_policy_params = sum(p.size for p in jax.tree_util.tree_leaves(policy_params))
    logging.info(f"num policy params = {num_policy_params}")

    if config.policy.anneal_lr:
        lr = optax.cosine_decay_schedule(
            config.policy.lr, decay_steps=num_update_steps, alpha=0.95
        )
    else:
        lr = config.policy.lr

    tx_policy = optax.chain(
        # optax.clip_by_global_norm(self.config.policy.max_grad_norm),
        optax.adam(lr, eps=config.policy.eps),
    )

    policy_apply = partial(
        jax.jit(
            policy_fn.apply,
            static_argnames=("config", "is_continuous", "action_dim"),
        ),
        config=FrozenConfigDict(config.policy),
        is_continuous=continuous_actions,
        action_dim=action_dim,
    )
    ts_policy = TrainState.create(
        apply_fn=policy_apply,
        params=policy_params,
        tx=tx_policy,
    )
    return ts_policy, policy_state


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

        self.ts_policy, self.policy_state = create_ts(
            config,
            next(self.rng_seq),
            self.envs,
            self.continuous_actions,
            self.action_dim,
            self.env_params.task_dim,
            num_update_steps=self.num_updates,
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
            task_dim=self.env_params.task_dim,
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
        task = xtimestep.timestep.state.goal

        if len(state.shape) == 1:  # add extra dimension
            state = state[..., np.newaxis]

        if len(task.shape) == 1:
            task = task[np.newaxis]

        self.policy_storage.prev_state[0] = state

        for step in range(self.steps_per_rollout):
            # sample random action from policy
            policy_output, self.policy_state = self.ts_policy.apply_fn(
                self.ts_policy.params,
                self.policy_state,
                next(self.rng_seq),
                env_state=state,
                task=task,
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
                masks=masks_done,
                bad_masks=masks_done,  # TODO: fix this?
                value_preds=value,
                done=done,
            )

            # update state
            state = next_state
            self.total_steps += self.num_processes

        # compute next value for bootstrapping
        policy_output, self.policy_state = self.ts_policy.apply_fn(
            self.ts_policy.params,
            self.policy_state,
            next(self.rng_seq),
            env_state=state,
            task=task,
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

        # import ipdb

        # ipdb.set_trace()
        policy_metrics, self.ts_policy, self.policy_state = update_policy(
            ts=self.ts_policy,
            state=self.policy_state,
            replay_buffer=self.policy_storage,
            config=FrozenConfigDict(self.config.policy),
            rng_key=next(self.rng_seq),
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
                state=self.policy_state,
                env=self.eval_envs,
                config=self.config,
                ts_policy=self.ts_policy,
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
                        state=self.policy_state,
                        env=self.eval_envs,
                        config=self.config,
                        ts_policy=self.ts_policy,
                        steps_per_rollout=self.steps_per_rollout,
                        action_dim=self.input_action_dim,
                        wandb_run=self.wandb_run,
                    )
                    if self.wandb_run is not None:
                        eval_metrics = gutl.prefix_dict_keys(eval_metrics, "eval/")
                        self.wandb_run.log(eval_metrics, step=self.iter_idx)

                if ((self.iter_idx + 1) % self.config.save_interval) == 0:
                    # save to pickle
                    ckpt_file = Path(self.ckpt_dir) / f"ckpt_{self.iter_idx}.pkl"
                    logging.debug(f"saving checkpoint to {ckpt_file}")
                    with open(ckpt_file, "wb") as f:
                        pickle.dump(
                            {
                                "config": self.config.to_dict(),
                                "ts_policy": self.ts_policy.params,
                                "policy_state": self.policy_state,
                            },
                            f,
                        )

            # Clean up after update
            self.policy_storage.after_update()

        self.envs.close()
        logging.info("Finished training, closing everything")
        return
