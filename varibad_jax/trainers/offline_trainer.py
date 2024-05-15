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
    create_data_loader,
    Batch,
    subsample_data,
)

from varibad_jax.models.bc.bc import BCAgent
from varibad_jax.models.decision_transformer.dt import (
    DecisionTransformerAgent,
    LatentDTAgent,
)
from varibad_jax.models.lam.lam import (
    LatentActionModel,
    LatentActionAgent,
    LatentActionDecoder,
)
from varibad_jax.models.vpt.vpt import VPT, VPTAgent


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

        self.train_dataset, self.eval_dataset = load_data(
            config=config, rng=rng, steps_per_rollout=steps_per_rollout
        )
        # these are trajectories
        self.eval_prompts = self.sample_prompts()

        # if config.data.data_type == "transitions":
        #     fn = lambda x: einops.rearrange(x, "B T ... -> (B T) ...")
        #     self.train_dataset = jtu.tree_map(fn, self.train_dataset)
        #     self.eval_dataset = jtu.tree_map(fn, self.eval_dataset)

        # import ipdb

        # ipdb.set_trace()

        if self.config.env.env_name == "procgen":
            self.train_dataloader = self.train_dataset.get_iter(self.config.batch_size)
            self.eval_dataloader = self.eval_dataset.get_iter(self.config.batch_size)
            # number of batches ot iterate over each epoch of training
            self.num_train_batches = 50
            self.num_eval_batches = 2
        else:
            self.train_dataloader = create_data_loader(
                self.train_dataset, data_cfg=config.data
            )
            logging.info(f"Number of train batches: {len(self.train_dataloader)}")

            if len(self.eval_dataset["observations"]) > 0:
                self.eval_dataloader = create_data_loader(
                    self.eval_dataset, data_cfg=config.data
                )
                logging.info(f"Number of eval batches: {len(self.eval_dataloader)}")
            else:
                self.eval_dataloader = None

            obs_shape = self.train_dataset["observations"].shape[1:]

        # test dataloader
        batch = next(iter(self.train_dataloader))

        fn = lambda x, y: logging.info(f"{jax.tree_util.keystr(x)}: {y.shape}")
        jtu.tree_map_with_path(fn, batch)

        if self.config.model.name == "bc":
            model_cls = BCAgent
        elif self.config.model.name == "dt_agent":
            model_cls = DecisionTransformerAgent
        elif self.config.model.name == "dt_lam_agent":
            model_cls = LatentDTAgent
        elif self.config.model.name == "lam":
            model_cls = LatentActionModel
        elif self.config.model.name == "latent_action_decoder":
            model_cls = LatentActionDecoder
        elif self.config.model.name == "lam_agent":
            model_cls = LatentActionAgent
        elif self.config.model.name == "vpt":
            model_cls = VPT
        elif self.config.model.name == "vpt_bc":
            model_cls = VPTAgent
        else:
            raise ValueError(f"Unknown model name: {self.config.model.name}")

        self.model = model_cls(
            config=config.model,
            observation_shape=obs_shape,  # determine obs shape from the dataset
            action_dim=self.action_dim,
            input_action_dim=self.input_action_dim,
            continuous_actions=self.continuous_actions,
            task_dim=self.config.env.task_dim,
            key=next(self.rng_seq),
        )

        if self.config.num_evals != -1:
            self.eval_every = int(self.config.num_epochs // self.config.num_evals)
        elif self.config.eval_perc != -1:
            self.eval_every = int(self.config.num_epochs * self.config.eval_perc)
        else:
            self.eval_every = self.config.eval_interval

        logging.info(f"evaluating model every: {self.eval_every}")

    def sample_prompts(self):
        eval_prompts = None
        if self.config.data.holdout_tasks or self.config.model.name == "dt_lam_agent":
            if (
                len(self.eval_dataset["observations"]) == 0
            ):  # use full training set if no eval
                num_trajs = self.train_dataset["observations"].shape[0]
                sample_dataset = self.train_dataset
            else:
                # sample a couple of trajectories from the evaluation dataset for inference
                num_trajs = self.eval_dataset["observations"].shape[0]
                sample_dataset = self.eval_dataset

            if num_trajs < self.config.num_eval_rollouts:
                replace = True
            else:
                replace = False

            prompt_idxs = npr.choice(
                num_trajs, size=self.config.num_eval_rollouts, replace=replace
            )
            eval_prompts = subsample_data(sample_dataset, prompt_idxs)

        return eval_prompts

    def train(self):
        # first eval
        if not self.config.skip_first_eval:
            eval_metrics = self.eval(epoch=0)

            if self.wandb_run is not None:
                metrics = gutl.prefix_dict_keys(eval_metrics, prefix="eval/")
                self.wandb_run.log(metrics)

        # train
        for epoch in tqdm.tqdm(range(self.config.num_epochs), desc="epochs"):
            # iterate over batches of data
            start_time = time.time()
            epoch_metrics = dd(list)
            for batch in self.train_dataloader:
                if "info" in batch:
                    del batch["info"]
                batch = Batch(**batch)
                metrics = self.model.update(next(self.rng_seq), batch)
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

            for k, v in epoch_metrics.items():
                epoch_metrics[k] = jnp.mean(jnp.array(v))

            epoch_time = time.time() - start_time
            epoch_metrics["time/epoch"] = epoch_time
            epoch_metrics["misc/lr"] = self.model._ts.opt_state.hyperparams["lr"]

            if ((epoch + 1) % self.eval_every) == 0:
                eval_metrics = self.eval(epoch + 1)

                print("train: ", epoch_metrics)
                print("eval: ", eval_metrics)

                if self.wandb_run is not None:
                    eval_metrics = gutl.prefix_dict_keys(eval_metrics, prefix="eval/")
                    self.wandb_run.log(eval_metrics, step=epoch + 1, commit=False)

            if self.wandb_run is not None:
                metrics = gutl.prefix_dict_keys(epoch_metrics, prefix="train/")
                self.wandb_run.log(metrics, step=epoch + 1)

        self.eval(epoch=self.config.num_epochs)

    def eval(self, epoch: int):
        eval_metrics = dd(list)

        # run on eval batches
        if self.eval_dataloader is not None:
            for batch in self.eval_dataloader:
                if "info" in batch:
                    del batch["info"]

                batch = Batch(**batch)
                metrics = self.model.update(
                    next(self.rng_seq), batch, update_model=False
                )
                for k, v in metrics.items():
                    eval_metrics[k].append(v)

            for k, v in eval_metrics.items():
                eval_metrics[k] = jnp.mean(jnp.array(v))

        if self.config.data.resample_prompts_every_eval:
            logging.info("getting new set of eval prompts")
            # get new set of prompts for evaluation
            self.eval_prompts = self.sample_prompts()

        # run rollouts
        if hasattr(self.model, "get_action") and self.config.run_eval_rollouts:
            rollout_metrics, *_ = run_rollouts(
                rng=next(self.rng_seq),
                agent=self.model,
                env=self.eval_envs,
                config=self.config,
                action_dim=self.model.input_action_dim,
                steps_per_rollout=self.steps_per_rollout,
                wandb_run=self.wandb_run,
                prompts=self.eval_prompts,
            )
            eval_metrics.update(rollout_metrics)

        # save model
        if self.config.mode == "train":
            self.save_model(self.model.save_dict, eval_metrics, epoch)

        return eval_metrics
