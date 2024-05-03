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
import tqdm
from functools import partial
from collections import defaultdict as dd
from varibad_jax.utils.rollout import run_rollouts

from varibad_jax.trainers.base_trainer import BaseTrainer
import varibad_jax.utils.general_utils as gutl
from varibad_jax.utils.data_utils import (
    load_data,
    create_traj_loader,
    create_lapo_loader,
)

from varibad_jax.models.decision_transformer.dt import DecisionTransformerAgent
from varibad_jax.models.lam.lam import (
    LatentActionModel,
    LatentActionAgent,
    LatentActionDecoder,
)


class OfflineTrainer(BaseTrainer):
    def __init__(self, config: ConfigDict):
        super().__init__(config)

        rng = npr.RandomState(config.seed)

        try:
            steps_per_rollout = (
                config.env.num_episodes_per_rollout * self.envs.max_episode_steps
            )
            self.steps_per_rollout = steps_per_rollout
        except:
            steps_per_rollout = None

        train_dataset, eval_dataset = load_data(
            config=config, rng=rng, steps_per_rollout=steps_per_rollout
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        if self.config.env.env_name == "procgen":
            self.train_dataloader = train_dataset.get_iter(self.config.batch_size)
            self.eval_dataloader = eval_dataset.get_iter(self.config.batch_size)
            # number of batches ot iterate over each epoch of training
            self.num_train_batches = 50
            self.num_eval_batches = 2
        else:
            loader = partial(
                create_traj_loader, num_traj_per_batch=config.num_traj_per_batch
            )

            if "lapo" in self.config.exp_name:
                loader = create_lapo_loader

            self.train_dataloader, self.num_train_batches = loader(
                train_dataset, rng, batch_size=config.batch_size
            )
            self.eval_dataloader, self.num_eval_batches = loader(
                eval_dataset, rng, batch_size=config.batch_size
            )

            logging.info(
                f"num_train_batches: {self.num_train_batches}, num_eval_batches: {self.num_eval_batches}"
            )

        # test dataloader
        batch = next(self.train_dataloader)
        for k, v in batch.items():
            logging.info(f"{k}: {v.shape}")

        # batch = next(self.eval_dataloader)
        # for k, v in batch.items():
        #     logging.info(f"{k}: {v.shape}")

        if self.config.model.name == "dt":
            model_cls = DecisionTransformerAgent
        elif self.config.model.name == "lam":
            model_cls = LatentActionModel
        elif self.config.model.name == "latent_action_decoder":
            model_cls = LatentActionDecoder
        elif self.config.model.name == "lam_agent":
            model_cls = LatentActionAgent
        else:
            raise ValueError(f"Unknown model name: {self.config.model.name}")

        self.model = model_cls(
            config=config.model,
            observation_shape=self.obs_shape,
            action_dim=self.action_dim,
            input_action_dim=self.input_action_dim,
            continuous_actions=self.continuous_actions,
            task_dim=self.config.env.task_dim,
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

            for k, v in epoch_metrics.items():
                epoch_metrics[k] = jnp.mean(jnp.array(v))

            epoch_time = time.time() - start_time
            epoch_metrics["time/epoch"] = epoch_time
            epoch_metrics["misc/lr"] = self.model._opt_state.hyperparams["lr"]

            if self.wandb_run is not None:
                metrics = gutl.prefix_dict_keys(epoch_metrics, prefix="train/")
                self.wandb_run.log(metrics)

            if (epoch + 1) % self.config.eval_interval == 0:
                eval_metrics = self.eval(epoch + 1)

                print(epoch_metrics)

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

        if self.config.model.name == "dt" and self.config.model.demo_conditioning:
            # sample a couple of demo prompts
            num_trajs = self.eval_dataset["observations"].shape[0]
            prompt_idxs = npr.choice(
                num_trajs, size=self.config.num_eval_rollouts, replace=False
            )
            prompts = {k: v[prompt_idxs] for k, v in self.eval_dataset.items()}
        else:
            prompts = None

        # run rollouts
        if hasattr(self.model, "get_action") and self.config.run_eval_rollouts:
            rollout_metrics, *_ = run_rollouts(
                rng=next(self.rng_seq),
                agent=self.model,
                env=self.eval_envs,
                config=self.config,
                action_dim=self.input_action_dim,
                steps_per_rollout=self.steps_per_rollout,
                wandb_run=self.wandb_run,
                prompts=prompts,
            )
            eval_metrics.update(rollout_metrics)

        # save model
        if self.config.mode == "train":
            self.save_model(self.model.save_dict, eval_metrics, epoch)

        return eval_metrics
