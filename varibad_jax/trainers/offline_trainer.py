from absl import logging
import jax
import chex
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
import einops
from collections import defaultdict as dd
from varibad_jax.utils.rollout import run_rollouts

from varibad_jax.trainers.base_trainer import BaseTrainer
import varibad_jax.utils.general_utils as gutl

from varibad_jax.models.decision_transformer.dt import DecisionTransformerAgent
from varibad_jax.models.lapo.lapo import LAPOModel, LAPOAgent, LAPOActionDecoder


@chex.dataclass
class Batch:
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray


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

        num_trajs, num_timesteps = data["observations"].shape[:2]
        avg_return = jnp.mean(jnp.sum(data["rewards"], axis=-1))
        logging.info(f"average return: {avg_return}")

        # need to convert rewards into returns to go
        # do i need to do discounting here?
        if self.config.model.name == "dt":
            returns = jnp.cumsum(data["rewards"][:, ::-1], axis=1)[:, ::-1]
            data["rewards"] = returns

        # [B, T, *], trajectory data
        if self.config.num_trajs != -1:
            indices = rng.choice(num_trajs, size=self.config.num_trajs, replace=False)
            for k, v in data.items():
                data[k] = v[indices]

        if self.config.model.name == "lapo_action_decoder":
            # flatten our data to be [B*T, *]
            for k, v in data.items():
                data[k] = einops.rearrange(v, "B T ... -> (B T) ...")

        for k, v in data.items():
            logging.info(f"{k}: {v.shape}")

        dataset_size = data["observations"].shape[0]
        num_train = int(dataset_size * self.config.train_frac)
        num_eval = dataset_size - num_train
        batch_size = self.config.batch_size

        num_complete_batches, leftover = divmod(dataset_size, batch_size)
        logging.info(f"num batches per episode: {num_complete_batches}")

        # split into train and eval
        train_data = {k: v[:num_train] for k, v in data.items()}
        eval_data = {k: v[num_train:] for k, v in data.items()}

        def create_traj_loader(data):
            num_complete_batches, leftover = divmod(
                data["observations"].shape[0], batch_size
            )
            num_batches = num_complete_batches + bool(leftover)

            def data_stream():
                while True:
                    perm = rng.permutation(num_train)
                    for i in range(num_batches):
                        batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                        batch = Batch(**{k: v[batch_idx] for k, v in data.items()})
                        yield batch

            batches = data_stream()
            return batches, num_batches

        def create_lapo_loader(data):
            # sample small windows of transitions in a trajectory
            num_complete_batches, leftover = divmod(
                data["observations"].shape[0], batch_size
            )
            num_batches = num_complete_batches + bool(leftover)

            def data_stream():
                while True:
                    perm = rng.permutation(num_train)
                    for i in range(num_batches):
                        batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                        time_idx = rng.randint(1, num_timesteps - 1)
                        batch = Batch(
                            **{
                                k: v[batch_idx, time_idx - 1 : time_idx + 2]
                                for k, v in data.items()
                            }
                        )
                        yield batch

            batches = data_stream()
            return batches, num_batches

        if (
            self.config.model.name == "dt"
            or self.config.model.name == "lapo_action_decoder"
        ):
            loader = create_traj_loader
        elif (
            self.config.model.name == "lapo"
            or self.config.model.name == "lapo_bc_agent"
        ):
            loader = create_lapo_loader

        self.train_dataloader, self.num_train_batches = loader(train_data)
        self.eval_dataloader, self.num_eval_batches = loader(eval_data)

        logging.info(
            f"len train dataset: {len(train_data['observations'])}, num train batches: {self.num_train_batches}"
        )
        logging.info(
            f"len eval dataset: {len(eval_data['observations'])}, num eval batches: {self.num_eval_batches}"
        )

        # test dataloader
        batch = next(self.train_dataloader)
        logging.info(f"batch observations: {batch.observations.shape}")
        logging.info(f"len batch: {len(batch)}")

        if self.config.model.name == "dt":
            model_cls = DecisionTransformerAgent
        elif self.config.model.name == "lapo":
            model_cls = LAPOModel
        elif self.config.model.name == "lapo_bc_agent":
            model_cls = LAPOAgent
        elif self.config.model.name == "lapo_action_decoder":
            model_cls = LAPOActionDecoder

        self.model = model_cls(
            config=config.model,
            observation_shape=train_data["observations"].shape[2:],
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
                metrics = self.model.update(next(self.rng_seq), batch)
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
            metrics = self.model.update(
                next(self.rng_seq), next(self.eval_dataloader), update_model=False
            )
            for k, v in metrics.items():
                eval_metrics[k].append(v)

        for k, v in eval_metrics.items():
            eval_metrics[k] = jnp.mean(jnp.array(v))

        # run rollouts
        if self.config.model.name == "dt":
            rollout_metrics, *_ = run_rollouts(
                rng=next(self.rng_seq),
                agent=self.model,
                env=self.eval_envs,
                config=self.config,
                action_dim=self.input_action_dim,
                steps_per_rollout=self.steps_per_rollout,
                wandb_run=self.wandb_run,
            )
            eval_metrics.update(rollout_metrics)

        # save model
        self.save_model(self.model.save_dict, eval_metrics, epoch)
        return eval_metrics
