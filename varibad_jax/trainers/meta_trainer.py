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
import wandb
import einops
import jax.tree_util as jtu

from varibad_jax.trainers.base_trainer import BaseTrainer
from varibad_jax.models.varibad.varibad import VariBADModel
from varibad_jax.agents.ppo.ppo import PPOAgent

from varibad_jax.utils.replay_buffer import OnlineStorage
from varibad_jax.utils.replay_vae import RolloutStorageVAE
import varibad_jax.utils.general_utils as gutl
from varibad_jax.utils.rollout import run_rollouts


class MetaRLTrainer(BaseTrainer):
    def __init__(self, config: ConfigDict):
        super().__init__(config)
        self.total_steps = 0
        self.num_processes = self.config.env.num_processes

        self.num_updates = (
            int(config.num_frames) // self.steps_per_rollout // self.num_processes
        )
        logging.info(f"num rl updates = {self.num_updates}")
        logging.info(f"steps per rollout = {self.steps_per_rollout}")
        logging.info(
            f"input action dim = {self.input_action_dim}, action_dim = {self.action_dim}"
        )

        self.belief_model = VariBADModel(
            config=config.vae,
            observation_shape=self.envs.observation_space.shape,
            action_dim=self.action_dim,
            input_action_dim=self.input_action_dim,
            continuous_actions=self.continuous_actions,
            key=next(self.rng_seq),
        )

        self.agent = PPOAgent(
            config=config.policy,
            observation_shape=self.envs.observation_space.shape,
            action_dim=self.action_dim,
            input_action_dim=self.input_action_dim,
            continuous_actions=self.continuous_actions,
            key=next(self.rng_seq),
        )
        self.policy_storage, self.vae_storage = self.setup_replay_buffers()

    def setup_replay_buffers(self):
        policy_storage = OnlineStorage(
            args=self.config,
            num_steps=self.steps_per_rollout,
            num_processes=self.num_processes,
            state_dim=self.envs.observation_space.shape,
            latent_dim=self.config.vae.latent_dim * 2,
            belief_dim=0,
            task_dim=0,
            action_space=self.envs.action_space,
            hidden_size=(
                self.config.vae.encoder.lstm_hidden_size
                if self.config.vae.encoder.name == "lstm"
                else self.config.vae.encoder.hidden_dim
            ),
            normalise_rewards=self.config.env.normalize_rews,
        )

        vae_storage = RolloutStorageVAE(
            num_processes=self.num_processes,
            max_trajectory_len=self.steps_per_rollout,
            zero_pad=True,
            max_num_rollouts=self.config.vae.buffer_size,
            state_dim=self.envs.observation_space.shape,
            action_dim=self.input_action_dim,
            task_dim=None,
            vae_buffer_add_thresh=1.0,
        )
        return policy_storage, vae_storage

    def collect_rollouts(self):
        logging.debug("inside rollout")

        reset_rng = next(self.rng_seq)
        reset_rng = jax.random.split(reset_rng, self.num_processes)
        xtimestep = self.jit_reset(self.env_params, reset_rng)
        state = xtimestep.timestep.observation
        state = state.astype(np.float32)
        if len(state.shape) == 1:  # add extra dimension
            state = state[..., np.newaxis]

        prior_outputs = self.belief_model.get_prior(
            next(self.rng_seq), batch_size=self.num_processes
        )

        latent_sample = prior_outputs.latent_sample
        latent_mean = prior_outputs.latent_mean
        latent_logvar = prior_outputs.latent_logvar
        latent = jnp.concatenate([latent_mean, latent_logvar], axis=-1)
        hidden_state = prior_outputs.hidden_state

        self.policy_storage.prev_state[0] = state
        # self.policy_storage.hidden_states[0] = hidden_state.copy()
        self.policy_storage.latent_samples.append(latent_sample.copy())
        self.policy_storage.latent_mean.append(latent_mean.copy())
        self.policy_storage.latent_logvar.append(latent_logvar.copy())

        # if using a transformer, we need to keep track of the full history
        if self.config.vae.encoder.name == "transformer":
            states = np.zeros(
                (self.steps_per_rollout, self.num_processes, *self.obs_shape)
            )
            actions = np.zeros(
                (self.steps_per_rollout, self.num_processes, self.input_action_dim)
            )
            rewards = np.zeros((self.steps_per_rollout, self.num_processes, 1))
            masks = np.zeros((self.steps_per_rollout, self.num_processes))

        for step in range(self.steps_per_rollout):
            # sample random action from policy
            policy_output, self.agent._state = self.agent.get_action(
                next(self.rng_seq),
                env_state=state,
                latent=latent,
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
                action = action[..., np.newaxis]

            done = done[:, np.newaxis]
            reward = rew_raw[:, np.newaxis]
            reward_norm = rew_norm[:, np.newaxis]
            value = policy_output.value

            # after seeing the next observation and getting more
            # information, we can update the belief over the task variable
            belief_model_input = dict(
                states=next_state,
                actions=action,
                rewards=reward,
                hidden_state=hidden_state,
                is_training=True,
            )
            if self.config.vae.encoder.name == "lstm":
                encode_outputs, self.belief_model._state = (
                    self.belief_model.encode_trajectory(
                        next(self.rng_seq), **belief_model_input
                    )
                )
            elif self.config.vae.encoder.name == "transformer":
                states[step] = next_state
                actions[step] = action
                rewards[step] = reward
                masks[step] = 1.0 - done.flatten()

                # we want to embed the full sequence cause jax
                encode_outputs, self.vae_state = self.ts_vae.apply_fn(
                    self.ts_vae.params,
                    self.vae_state,
                    next(self.rng_seq),
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    mask=masks,
                    is_training=True,
                )

                # take the last timestep for every item
                encode_outputs.latent_mean = encode_outputs.latent_mean[step]
                encode_outputs.latent_logvar = encode_outputs.latent_logvar[step]
                encode_outputs.latent_sample = encode_outputs.latent_sample[step]

            # compute reward bonuses with RND / HyperX
            if self.config.policy.use_hyperx_bonuses:
                # hyper-state bonus
                rew_rnd, _ = compute_hyperx_bonus(
                    self.ts_hyperx_predictor.params,
                    self.ts_hyperx_prior,
                    self.ts_hyperx_predictor,
                    next(self.rng_seq),
                    obs=next_state,
                    latent=latent,
                )
                rew_rnd = rew_rnd[:, np.newaxis]

                # vae reconstruction bonus
                decode_outputs, _ = self.decode_apply(
                    self.ts_vae.params,
                    self.vae_state,
                    next(self.rng_seq),
                    latent_samples=encode_outputs.latent_sample,
                    prev_states=state,
                    next_states=next_state,
                    actions=action,
                )
                rew_recon_err = optax.squared_error(decode_outputs.rew_pred, reward)

                # import ipdb; ipdb.set_trace()
                rew_rnd *= self.config.hyperx.rnd_weight
                rew_recon_err *= self.config.hyperx.vae_recon_weight
            else:
                rew_rnd = None
                rew_recon_err = None

            # add transition to vae buffer
            self.vae_storage.insert(
                prev_state=state,
                next_state=next_state,
                actions=action,
                rewards=reward,
                done=done,
                task=None,
            )
            # print("*" * 20)
            # print(step, next_state)
            # print(xtimestep.timestep.state.goal)
            # print(xtimestep.info["done"])
            # print(xtimestep.info["done_bamdp"])
            # print(xtimestep.timestep.last())
            # print(xtimestep.info["step_count_bamdp"])
            # print("-" * 20)
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
                hyperx_bonuses=rew_rnd,
                vae_recon_bonuses=rew_recon_err,
                masks=masks_done,
                bad_masks=masks_done,  # TODO: fix this?
                value_preds=value,
                done=done,
                # hidden_states=encode_outputs.hidden_state,
                latent_mean=encode_outputs.latent_mean,
                latent_logvar=encode_outputs.latent_logvar,
                latent_sample=encode_outputs.latent_sample,
            )

            # update state
            state = next_state
            latent_mean = encode_outputs.latent_mean
            latent_logvar = encode_outputs.latent_logvar
            latent = jnp.concatenate([latent_mean, latent_logvar], axis=-1)
            hidden_state = encode_outputs.hidden_state
            self.total_steps += self.num_processes

        # compute next value for bootstrapping
        policy_output, self.agent._state = self.agent.get_action(
            next(self.rng_seq),
            env_state=state,
            latent=latent,
            is_training=True,
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

        # import ipdb

        # ipdb.set_trace()
        logging.debug("done collecting rollouts")

    def perform_update(self):
        logging.debug("inside perform update")
        metrics = {}

        # compute ppo update
        update_start = time.time()

        # import ipdb

        # ipdb.set_trace()
        policy_metrics = self.agent.update(
            replay_buffer=self.policy_storage,
            rng=next(self.rng_seq),
        )
        policy_update_time = time.time() - update_start
        policy_metrics = gutl.prefix_dict_keys(policy_metrics, prefix="policy/")
        metrics.update(policy_metrics)
        logging.debug("done updating policy")

        # update vae
        update_start = time.time()
        for _ in range(self.config.vae.num_vae_updates):
            vae_metrics = self.belief_model.update(
                rng=next(self.rng_seq),
                batch=self.vae_storage.get_batch(self.config.vae.trajs_per_batch),
            )
        vae_update_time = time.time() - update_start
        vae_metrics = gutl.prefix_dict_keys(vae_metrics, prefix="vae/")
        metrics.update(vae_metrics)
        metrics["time/vae_update_time"] = vae_update_time
        metrics["time/policy_update_time"] = policy_update_time

        logging.debug("done updating vae")

        # update hyperx bonus models
        if self.config.policy.use_hyperx_bonuses:
            self.ts_hyperx_predictor, (hyperx_loss, _) = update_hyperx(
                ts_predictor=self.ts_hyperx_predictor,
                ts_prior=self.ts_hyperx_prior,
                ts_vae=self.ts_vae,
                state=self.hyperx_state,
                vae_state=self.vae_state,
                config=FrozenConfigDict(self.config),
                batch=self.vae_storage.get_batch(self.config.vae.trajs_per_batch),
                rng_key=next(self.rng_seq),
                get_prior_fn=self.get_prior,
            )
        return metrics

    def train(self):
        logging.info("Training starts")
        start_time = time.time()

        # first eval
        if not self.config.skip_first_eval:
            (
                eval_metrics,
                *_,
            ) = run_rollouts(
                rng=next(self.rng_seq),
                agent=self.agent,
                belief_model=self.belief_model,
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
                        "rew_avg": self.policy_storage.rewards_raw.mean(),
                        "rew_sum": self.policy_storage.rewards_raw.sum(),
                        "rew_goal": (self.policy_storage.rewards_raw == 1).sum(),
                    }
                    env_stats = gutl.prefix_dict_keys(env_stats, prefix="env/")
                    encoder_stats = {
                        "latent_mean": np.array(self.policy_storage.latent_mean).mean(),
                        "latent_logvar": np.array(
                            self.policy_storage.latent_logvar
                        ).mean(),
                    }
                    encoder_stats = gutl.prefix_dict_keys(
                        encoder_stats, prefix="encoder/"
                    )
                    metrics = {
                        **train_metrics,
                        **env_stats,
                        **encoder_stats,
                        "misc/agent_lr": self.agent._opt_state.hyperparams["lr"],
                        "misc/belief_lr": self.belief_model._opt_state.hyperparams[
                            "lr"
                        ],
                        "time/rollout_time": rollout_time,
                        "time/fps": self.total_steps / (time.time() - start_time),
                    }
                    if self.wandb_run is not None:
                        self.wandb_run.log(metrics, step=self.iter_idx)

                if (self.iter_idx + 1) % self.config.eval_interval == 0:
                    eval_metrics, *_ = run_rollouts(
                        rng=next(self.rng_seq),
                        agent=self.agent,
                        belief_model=self.belief_model,
                        env=self.eval_envs,
                        config=self.config,
                        steps_per_rollout=self.steps_per_rollout,
                        action_dim=self.input_action_dim,
                        wandb_run=self.wandb_run,
                    )

                    save_dict = {
                        "agent": self.agent.save_dict,
                        "belief_model": self.belief_model.save_dict,
                    }

                    self.save_model(save_dict, eval_metrics, iter_idx=self.iter_idx)

                    if self.wandb_run is not None:
                        eval_metrics = gutl.prefix_dict_keys(eval_metrics, "eval/")
                        self.wandb_run.log(eval_metrics, step=self.iter_idx)

            # Clean up after update
            self.policy_storage.after_update()

        self.envs.close()
        logging.info("Finished training, closing everything")
        return
