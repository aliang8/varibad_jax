from absl import logging
import jax
import time
import pickle
from ml_collections import ConfigDict
import numpy.random as npr
from pathlib import Path
import jax.tree_util as jtu
import jax.numpy as jnp
import haiku as hk
import optax
import functools
import gymnasium as gym
import tqdm
import numpy as np
from collections import defaultdict as dd
from flax.training.train_state import TrainState
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
from varibad_jax.utils.rollout import run_rollouts
from typing import NamedTuple

from varibad_jax.trainers.base_trainer import BaseTrainer
import varibad_jax.utils.general_utils as gutl
from varibad_jax.models.decision_transformer.helpers import init_params_dt, dt_fn
from varibad_jax.models.decision_transformer.loss import loss_fn as dt_loss_fn

from varibad_jax.models.genie.helpers import init_params_lam, lam_apply_fn
from varibad_jax.models.genie.loss import loss_fn as lam_loss_fn


def create_ts(
    config: ConfigDict, rng, envs, continuous_actions, input_action_dim, action_dim
):
    if config.policy.name == "dt":
        params = init_params_dt(
            config.policy,
            rng,
            envs.observation_space,
            continuous_actions,
            input_action_dim,
            action_dim,
        )
        policy_apply = functools.partial(
            jax.jit(
                dt_fn.apply,
                static_argnames=("config", "action_dim", "is_continuous"),
            ),
            config=FrozenConfigDict(config.policy),
            action_dim=action_dim,
            is_continuous=continuous_actions,
        )
        state = None
    elif config.policy.name == "lam":
        params, state = init_params_lam(
            config.policy,
            rng,
            envs.observation_space,
        )
        policy_apply = functools.partial(
            jax.jit(
                lam_apply_fn.apply,
                static_argnames=("config"),
            ),
            config=FrozenConfigDict(config.policy),
        )

    num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    logging.info(f"num params = {num_params}")

    tx = optax.chain(
        # optax.clip(config.vae.max_grad_norm),
        optax.adam(config.lr, eps=config.eps),
    )
    ts = TrainState.create(
        apply_fn=policy_apply,
        params=params,
        tx=tx,
    )
    return ts, state


class OfflineTrainer(BaseTrainer):
    def __init__(self, config: ConfigDict):
        super().__init__(config)

        self.ts_policy, self.state = create_ts(
            config,
            next(self.rng_seq),
            self.envs,
            self.continuous_actions,
            self.input_action_dim,
            self.action_dim,
        )

        rng = npr.RandomState(config.seed)

        steps_per_rollout = (
            config.env.num_episodes_per_rollout * self.envs.max_episode_steps
        )
        self.steps_per_rollout = steps_per_rollout
        data_file = f"eid-{config.env.env_id}_n-{config.num_rollouts_collect}_steps-{steps_per_rollout}.pkl"
        data_path = Path(self.config.root_dir) / self.config.data_dir / data_file
        with open(data_path, "rb") as f:
            timesteps, actions = pickle.load(f)

        dataset_size = actions.shape[0]
        num_train = int(dataset_size * self.config.train_frac)
        num_eval = dataset_size - num_train

        # make into list of pytrees
        @jax.jit
        def get_ts(i):
            return jtu.tree_map(lambda y: y[i], timesteps)

        observations = jnp.array([get_ts(i).observation for i in range(dataset_size)])
        rewards = jnp.array([get_ts(i).reward for i in range(dataset_size)])

        logging.info(
            f"observations shape: {observations.shape}, rewards shape: {rewards.shape}, actions shape: {actions.shape}"
        )
        logging.info(f"average return: {jnp.mean(jnp.sum(rewards, axis=-1))}")

        # need to convert rewards into returns to go
        # do i need to do discounting here?
        returns = jnp.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]

        # split into train and eval
        train_observations, train_actions, train_rewards = (
            observations[:num_train],
            actions[:num_train],
            returns[:num_train],
        )
        eval_observations, eval_actions, eval_rewards = (
            observations[num_train:],
            actions[num_train:],
            returns[num_train:],
        )

        batch_size = self.config.batch_size

        def create_loader(observations, actions, rewards):
            num_complete_batches, leftover = divmod(observations.shape[0], batch_size)
            num_batches = num_complete_batches + bool(leftover)

            def data_stream():
                while True:
                    perm = rng.permutation(num_train)
                    for i in range(num_batches):
                        batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                        yield observations[batch_idx], actions[batch_idx], rewards[
                            batch_idx
                        ]

            batches = data_stream()
            return batches, num_batches

        self.train_dataloader, self.num_train_batches = create_loader(
            train_observations, train_actions, train_rewards
        )
        self.eval_dataloader, self.num_eval_batches = create_loader(
            eval_observations, eval_actions, eval_rewards
        )

        logging.info(
            f"len train dataset: {len(train_observations)}, num train batches: {self.num_train_batches}"
        )
        logging.info(
            f"len eval dataset: {len(eval_observations)}, num eval batches: {self.num_eval_batches}"
        )

        if self.config.policy.name == "dt":
            loss_fn = functools.partial(
                dt_loss_fn,
                continuous_actions=self.continuous_actions,
            )
        elif self.config.policy.name == "lam":
            loss_fn = lam_loss_fn

        def update_step(ts, state, batch, rng):
            (loss, (metrics, state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                ts.params, ts, state, batch, rng
            )

            ts = ts.apply_gradients(grads=grads)
            return ts, metrics, state

        self.jit_update_step = jax.jit(update_step)

    def train(self):
        # first eval
        if not self.config.skip_first_eval:
            eval_metrics = self.eval()
            if self.wandb_run is not None:
                metrics = gutl.prefix_dict_keys(eval_metrics, prefix="eval/")
                self.wandb_run.log(metrics)

        # train
        for epoch in tqdm.tqdm(range(self.config.num_epochs)):
            # iterate over batches of data
            start_time = time.time()
            epoch_metrics = dd(list)
            for _ in range(self.num_train_batches):
                batch = next(self.train_dataloader)
                self.ts_policy, metrics, self.state = self.jit_update_step(
                    self.ts_policy, self.state, batch, next(self.rng_seq)
                )
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

            # average a list of dicts using jax tree operations
            for k, v in epoch_metrics.items():
                epoch_metrics[k] = jnp.mean(jnp.array(v))

            epoch_time = time.time() - start_time
            epoch_metrics["time/epoch"] = epoch_time

            if self.wandb_run is not None:
                metrics = gutl.prefix_dict_keys(epoch_metrics, prefix="train/")
                self.wandb_run.log(metrics)

            if (epoch + 1) % self.config.eval_interval == 0:
                eval_metrics = self.eval()
                if self.wandb_run is not None:
                    eval_metrics = gutl.prefix_dict_keys(eval_metrics, prefix="eval/")
                    self.wandb_run.log(eval_metrics)

    def eval(self):
        eval_metrics = dd(list)

        # run on eval batches
        for _ in range(self.num_eval_batches):
            loss, metrics = jax.jit(self.loss_fn)(
                self.ts.params,
                self.ts_policy,
                next(self.eval_dataloader),
                next(self.rng_seq),
            )
            for k, v in metrics.items():
                eval_metrics[k].append(v)

        for k, v in eval_metrics.items():
            eval_metrics[k] = jnp.mean(jnp.array(v))

        # run rollouts
        rollout_metrics = run_rollouts(
            rng=next(self.rng_seq),
            num_rollouts=self.config.num_eval_rollouts,
            env=self.eval_envs,
            config=self.config,
            ts_policy=self.ts_policy,
            action_dim=self.input_action_dim,
            steps_per_rollout=self.steps_per_rollout,
            wandb_run=self.wandb_run,
        )
        eval_metrics.update(rollout_metrics)
        return eval_metrics
