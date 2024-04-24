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
from varibad_jax.models.varibad.helpers import init_params as init_params_vae
from varibad_jax.models.varibad.helpers import encode_trajectory, decode, get_prior
from varibad_jax.models.varibad.update import update_vae
from varibad_jax.agents.ppo.ppo import update_policy
from varibad_jax.agents.ppo.helpers import init_params as init_params_policy
from varibad_jax.agents.ppo.helpers import policy_fn

from varibad_jax.models.hyperx.helpers import init_params_hyperx, hyperx_apply_fn, compute_hyperx_bonus
from varibad_jax.models.hyperx.update import update_hyperx


from varibad_jax.utils.replay_buffer import OnlineStorage
from varibad_jax.utils.replay_vae import RolloutStorageVAE
import varibad_jax.utils.general_utils as gutl
from varibad_jax.utils.rollout import run_rollouts

@optax.inject_hyperparams
def optimizer(lr, eps):
    return optax.chain(
        # optax.clip_by_global_norm(config.policy.max_grad_norm),
        optax.adam(lr, eps=eps),
    )

def create_ts(config, rng, envs, input_action_dim, action_dim, num_update_steps: int):
    continuous_actions = not isinstance(envs.action_space, gym.spaces.Discrete)

    if config.load_from_ckpt:
        model_ckpt_dir = Path(config.root_dir) / config.model_ckpt_dir
        ckpt_file = model_ckpt_dir / f"ckpt_{config.checkpoint_step}.pkl"

        with open(ckpt_file, "rb") as f:
            ckpt = pickle.load(f)
            vae_params = ckpt["ts_vae"]
            policy_params = ckpt["ts_policy"]
            vae_state = ckpt["vae_state"]
            policy_state = ckpt["policy_state"]
    else:
        rng_vae, rng_policy = jax.random.split(rng, 2)
        vae_params, vae_state = init_params_vae(
            config=config.vae,
            rng_key=rng_vae,
            observation_space=envs.observation_space,
            action_dim=input_action_dim,
        )
        policy_params, policy_state = init_params_policy(
            config=config.policy,
            rng_key=rng_policy,
            observation_space=envs.observation_space,
            latent_dim=config.vae.latent_dim * 2,
            action_space=envs.action_space,
        )

    # count number of vae params
    num_vae_params = sum(p.size for p in jax.tree_util.tree_leaves(vae_params))
    logging.info(f"num vae params = {num_vae_params}")

    tx_vae = optimizer(config.vae.lr, config.vae.eps) 
    
    # import ipdb

    # ipdb.set_trace()
    num_policy_params = sum(p.size for p in jax.tree_util.tree_leaves(policy_params))
    logging.info(f"num policy params = {num_policy_params}")

    if config.policy.anneal_lr:
        if config.lr_anneal_method == "cosine":
            lr = optax.cosine_decay_schedule(
                config.policy.lr, decay_steps=num_update_steps, alpha=0.95
            )
        elif config.lr_anneal_method == "warmup_exp_decay":
            lr = optax.warmup_exponential_decay_schedule(
                init_value=1e-5,
                peak_value=config.policy.lr,
                warmup_steps=int(num_update_steps * 0.1),
                decay_rate=0.8,
                transition_steps=int(num_update_steps * 0.1),
                end_value=1e-5,
            )
    else:
        lr = config.policy.lr

    tx_policy = optimizer(lr, config.policy.eps)
    encode_apply = partial(
        jax.jit(
            encode_trajectory.apply,
            static_argnames=("config", "is_training"),
        ),
        config=FrozenConfigDict(config.vae),
    )

    ts_vae = TrainState.create(apply_fn=encode_apply, params=vae_params, tx=tx_vae)

    policy_apply = partial(
        jax.jit(
            policy_fn.apply,
            static_argnames=("config", "is_continuous", "action_dim", "is_training"),
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
    return ts_vae, ts_policy, vae_state, policy_state

def create_hyperx_ts(config, rng, envs):
    hyperx_params, hyperx_state = init_params_hyperx(
        config=config.hyperx,
        rng=rng,
        observation_space=envs.observation_space,
    )
    num_hyperx_params = sum(p.size for p in jax.tree_util.tree_leaves(hyperx_params))
    logging.info(f"num rnd params = {num_hyperx_params}")

    tx_hyperx = optimizer(config.hyperx.lr, config.hyperx.eps)

    hyperx_apply = partial(
        jax.jit(
            hyperx_apply_fn.apply,
            static_argnames=("config"),
        ),
        config=FrozenConfigDict(config.hyperx.rnd),
    )

    ts_hyperx = TrainState.create(
        apply_fn=hyperx_apply,
        params=hyperx_params,
        tx=tx_hyperx,
    )

    return ts_hyperx, hyperx_state

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

        self.ts_vae, self.ts_policy, self.vae_state, self.policy_state = create_ts(
            config=config,
            rng=next(self.rng_seq),
            envs=self.envs,
            input_action_dim=self.input_action_dim,
            action_dim=self.action_dim,
            num_update_steps=self.num_updates,
        )
        self.decode_apply = partial(
            jax.jit(
                decode.apply,
                static_argnames=("config", "is_training"),
            ),
            config=FrozenConfigDict(config.vae),
        )

        self.get_prior = partial(
            jax.jit(
                get_prior.apply,
                static_argnames=("config", "batch_size"),
            ),
            config=FrozenConfigDict(config.vae),
        )

        if self.config.policy.use_hyperx_bonuses:
            # create ts for hyperx bonus
            # prior network is randomly initialized and not updated
            self.ts_hyperx_prior, self.hyperx_state = create_hyperx_ts(
                config=config,
                rng=next(self.rng_seq),
                envs=self.envs,
            )
            self.ts_hyperx_predictor, _ = create_hyperx_ts(
                config=config,
                rng=next(self.rng_seq),
                envs=self.envs,
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

        prior_outputs = self.get_prior(
            self.ts_vae.params, next(self.rng_seq), batch_size=self.num_processes
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
            policy_output, self.policy_state = self.ts_policy.apply_fn(
                self.ts_policy.params,
                self.policy_state,
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
            if self.config.vae.encoder.name == "lstm":
                encode_outputs, self.vae_state = self.ts_vae.apply_fn(
                    self.ts_vae.params,
                    self.vae_state,
                    next(self.rng_seq),
                    states=next_state,
                    actions=action,
                    rewards=reward,
                    hidden_state=hidden_state,
                    is_training=True,
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
                    self.ts_hyperx_predictor.params, self.ts_hyperx_prior, self.ts_hyperx_predictor, next(self.rng_seq), obs=next_state, latent=latent)
                rew_rnd = rew_rnd[:, np.newaxis]
                
                # vae reconstruction bonus 
                decode_outputs, _ = self.decode_apply(self.ts_vae.params, self.vae_state, next(self.rng_seq), latent_samples=encode_outputs.latent_sample, prev_states=state, next_states=next_state, actions=action)
                rew_recon_err = optax.squared_error(decode_outputs.rew_pred, reward)

                # import ipdb; ipdb.set_trace()
                rew_rnd *= self.config.hyperx.rnd_weight
                rew_recon_err *= self.config.hyperx.vae_recon_weight
            else:
                rew_rnd = None

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
        policy_output, self.policy_state = self.ts_policy.apply_fn(
            self.ts_policy.params,
            self.policy_state,
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

        # update vae
        update_start = time.time()
        for _ in range(self.config.vae.num_vae_updates):
            self.ts_vae, (_, vae_metrics, self.vae_state) = update_vae(
                ts=self.ts_vae,
                state=self.vae_state,
                rng_key=next(self.rng_seq),
                config=FrozenConfigDict(self.config),
                batch=self.vae_storage.get_batch(self.config.vae.trajs_per_batch),
                decode_fn=self.decode_apply,
                get_prior_fn=self.get_prior,
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
                state=[self.vae_state, self.policy_state],
                env=self.eval_envs,
                config=self.config,
                ts_policy=self.ts_policy,
                ts_vae=self.ts_vae,
                get_prior=self.get_prior,
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
                        "misc/lr": self.ts_policy.opt_state.hyperparams["lr"],
                        "time/rollout_time": rollout_time,
                        "time/fps": self.total_steps / (time.time() - start_time),
                    }
                    if self.wandb_run is not None:
                        self.wandb_run.log(metrics, step=self.iter_idx)

                if (self.iter_idx + 1) % self.config.eval_interval == 0:
                    eval_metrics, *_ = run_rollouts(
                        rng=next(self.rng_seq),
                        state=[self.vae_state, self.policy_state],
                        env=self.eval_envs,
                        config=self.config,
                        ts_policy=self.ts_policy,
                        ts_vae=self.ts_vae,
                        get_prior=self.get_prior,
                        steps_per_rollout=self.steps_per_rollout,
                        action_dim=self.input_action_dim,
                        wandb_run=self.wandb_run,
                    )
                    if self.wandb_run is not None:
                        eval_metrics = gutl.prefix_dict_keys(eval_metrics, "eval/")
                        self.wandb_run.log(eval_metrics, step=self.iter_idx)

                if (self.iter_idx + 1) % self.config.save_interval == 0:
                    # save to pickle
                    ckpt_file = Path(self.ckpt_dir) / f"ckpt_{self.iter_idx + 1}.pkl"
                    logging.debug(f"saving checkpoint to {ckpt_file}")
                    with open(ckpt_file, "wb") as f:
                        pickle.dump(
                            {
                                "config": self.config.to_dict(),
                                "ts_policy": self.ts_policy.params,
                                "ts_vae": self.ts_vae.params,
                                "vae_state": self.vae_state,
                                "policy_state": self.policy_state,
                            },
                            f,
                        )

            # Clean up after update
            self.policy_storage.after_update()

        self.envs.close()
        logging.info("Finished training, closing everything")
        return
