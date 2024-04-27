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
from varibad_jax.utils.rollout import run_rollouts

from varibad_jax.trainers.base_trainer import BaseTrainer
import varibad_jax.utils.general_utils as gutl

from varibad_jax.models.decision_transformer.dt import DecisionTransformerAgent
from varibad_jax.models.lapo.lapo import LAPOAgent


class OfflineTrainer(BaseTrainer):
    def __init__(self, config: ConfigDict):
        super().__init__(config)

        rng = npr.RandomState(config.seed)

        steps_per_rollout = (
            config.env.num_episodes_per_rollout * self.envs.max_episode_steps
        )
        self.steps_per_rollout = steps_per_rollout

        # load dataset
        dataset_name = f"eid-{config.env.env_id}_n-{config.num_rollouts_collect}_steps-{steps_per_rollout}"
        data_path = (
            Path(self.config.root_dir)
            / self.config.data_dir
            / dataset_name
            / "dataset.pkl"
        )
        logging.info(f"loading data from {data_path}")
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        # [B, T, *], trajectory data
        observations = data["observations"]
        next_observations = data["next_observations"]
        actions = data["actions"]
        rewards = data["rewards"]

        dataset_size = observations.shape[0]
        num_train = int(dataset_size * self.config.train_frac)
        num_eval = dataset_size - num_train

        logging.info(
            f"observations shape: {observations.shape}, rewards shape: {rewards.shape}, actions shape: {actions.shape}"
        )
        logging.info(f"average return: {jnp.mean(jnp.sum(rewards, axis=-1))}")

        # need to convert rewards into returns to go
        # do i need to do discounting here?
        if self.config.policy.name == "dt":
            returns = jnp.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
        else:
            returns = rewards

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

        # def create_loader(observations, actions, rewards):
        #     num_complete_batches, leftover = divmod(observations.shape[0], batch_size)
        #     num_batches = num_complete_batches + bool(leftover)

        #     def data_stream():
        #         while True:
        #             perm = rng.permutation(num_train)
        #             for i in range(num_batches):
        #                 batch_idx = perm[i * batch_size : (i + 1) * batch_size]
        #                 yield observations[batch_idx], actions[batch_idx], rewards[
        #                     batch_idx
        #                 ]

        #     batches = data_stream()
        #     return batches, num_batches

        def create_loader(observations, actions, rewards):
            num_complete_batches, leftover = divmod(observations.shape[0], batch_size)
            num_batches = num_complete_batches + bool(leftover)

            # this streamer is for LAPO training
            def data_stream():
                while True:
                    perm = rng.permutation(num_train)
                    for i in range(num_batches):
                        batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                        time_idx = rng.randint(1, observations.shape[1] - 1)
                        yield observations[
                            batch_idx, time_idx - 1 : time_idx + 2
                        ], actions[batch_idx, time_idx - 1 : time_idx + 2], rewards[
                            batch_idx, time_idx - 1 : time_idx + 2
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

        # test dataloader
        batch = next(self.train_dataloader)
        logging.info(f"batch observations: {batch[0].shape}")
        logging.info(f"len batch: {len(batch)}")

        if self.config.policy.name == "dt":
            agent_cls = DecisionTransformerAgent
        elif self.config.policy.name == "lapo":
            agent_cls = LAPOAgent

        self.agent = agent_cls(
            config=config.policy,
            observation_shape=observations.shape[2:],
            action_dim=self.action_dim,
            input_action_dim=self.input_action_dim,
            continuous_actions=self.continuous_actions,
            key=next(self.rng_seq),
        )

    def train(self):
        # first eval
        if not self.config.skip_first_eval:
            eval_metrics = self.eval(epoch=0)

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
                metrics = self.agent.update(next(self.rng_seq), batch)
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
                eval_metrics = self.eval(epoch + 1)

                if self.wandb_run is not None:
                    eval_metrics = gutl.prefix_dict_keys(eval_metrics, prefix="eval/")
                    self.wandb_run.log(eval_metrics)

    def eval(self, epoch: int):
        eval_metrics = dd(list)

        # run on eval batches
        for _ in range(self.num_eval_batches):
            metrics = self.agent.update(
                next(self.rng_seq), next(self.eval_dataloader), update_model=False
            )
            for k, v in metrics.items():
                eval_metrics[k].append(v)

        for k, v in eval_metrics.items():
            eval_metrics[k] = jnp.mean(jnp.array(v))

        # run rollouts
        rollout_metrics, *_ = run_rollouts(
            rng=next(self.rng_seq),
            agent=self.agent,
            env=self.eval_envs,
            config=self.config,
            action_dim=self.input_action_dim,
            steps_per_rollout=self.steps_per_rollout,
            wandb_run=self.wandb_run,
        )
        eval_metrics.update(rollout_metrics)

        # save model
        self.save_model(self.agent.save_dict, eval_metrics, epoch)
        return eval_metrics
