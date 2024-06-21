from absl import logging
import re
import json
import copy
import jax
import chex
import time
import pickle
from ml_collections import ConfigDict
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
import wandb
from functools import partial
from collections import defaultdict as dd
from varibad_jax.utils.rollout import run_rollouts
from varibad_jax.utils.rollout_procgen import run_rollouts_procgen
from varibad_jax.utils.rollout_atari import run_rollouts_atari
from varibad_jax.utils.visualization import custom_to_pil, plot_images, make_image_grid

from varibad_jax.trainers.base_trainer import BaseTrainer
import varibad_jax.utils.general_utils as gutl
from varibad_jax.utils.data_utils import (
    load_data,
    create_data_loader,
    Batch,
    subsample_data,
)
from varibad_jax.utils.tfds_data_utils import load_data as load_data_tfds

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
from varibad_jax.models.vpt.vpt import VPT, VPTAgent, VPTDTAgent


class OfflineTrainer(BaseTrainer):
    def __init__(self, config: ConfigDict):
        super().__init__(config)

        try:
            steps_per_rollout = (
                config.env.num_episodes_per_rollout * self.envs.max_episode_steps
            )
            self.steps_per_rollout = steps_per_rollout
        except:
            steps_per_rollout = None

        data_load_time = time.time()
        # self.train_dataset, self.eval_dataset, self.prompt_dataset = load_data_tfds(
        #     config=config, rng=next(self.rng_seq)
        # )

        # logging.info(f"data loading time: {time.time() - data_load_time}")

        # these are trajectories
        # self.prompts = self.sample_prompts()

        # self.eval_dataloaders = {}
        # self.train_dataloader = create_data_loader(
        #     self.train_dataset, data_cfg=config.data
        # )
        # logging.info(f"Number of train batches: {len(self.train_dataloader)}")
        # for eval_env_id, eval_dataset in self.eval_dataset.items():
        #     if len(eval_dataset["observations"]) > 0:
        #         eval_dataloader = create_data_loader(eval_dataset, data_cfg=config.data)
        #         logging.info(f"Number of eval batches: {len(eval_dataloader)}")
        #         self.eval_dataloaders[eval_env_id] = eval_dataloader

        # if config.data.data_type == "trajectories":
        #     obs_shape = self.train_dataset["observations"].shape[2:]
        # else:
        #     obs_shape = self.train_dataset["observations"].shape[1:]

        # update batch size
        config.data.batch_size = config.data.batch_size * config.num_xla_devices
        self.train_dataloader, self.eval_dataloaders, self.prompt_dataloader = (
            load_data_tfds(config=config, rng=next(self.rng_seq))
        )

        # print batch item shapes
        batch = next(iter(self.train_dataloader))
        if config.data.data_type in ["lapo", "trajectories"]:
            obs_shape = batch["observations"].shape[2:]
        else:
            obs_shape = batch["observations"].shape[1:]

        for k, v in batch.items():
            logging.info(f"{k}: {v.shape}")

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
        elif self.config.model.name == "vpt_icl_agent":
            model_cls = VPTDTAgent
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
            num_devices=min(self.config.num_xla_devices, self.num_devices),
        )

        if self.config.eval_interval != -1:
            self.eval_every = self.config.eval_interval
        elif self.config.num_evals != -1:
            self.eval_every = int(self.config.num_updates // self.config.num_evals)
        elif self.config.eval_perc != -1:
            self.eval_every = int(self.config.num_updates * self.config.eval_perc)
        else:
            raise ValueError("no eval interval specified")

        logging.info(f"evaluating model every: {self.eval_every}")

    def sample_prompts(self):
        assert self.prompt_dataset is not None

        prompts = {}
        if self.config.data.holdout_tasks or (
            "policy" in self.config.model and self.config.model.policy.demo_conditioning
        ):
            logging.info("sampling new prompts for evaluation")
            for env_id, prompt_dataset in self.prompt_dataset.items():
                data_cfg = copy.deepcopy(self.config.data)
                data_cfg.num_trajs_per_batch -= 1
                data_cfg.batch_size = self.config.num_eval_rollouts
                loader = create_data_loader(prompt_dataset, data_cfg)
                prompts[env_id] = next(iter(loader))

        return prompts

    def train(self):
        # first eval
        if not self.config.skip_first_eval:
            eval_metrics = self.eval(step=0)

        train_iter = self.train_dataloader.repeat().as_numpy_iterator()

        # train
        for train_step in tqdm.tqdm(
            range(self.config.num_updates),
            desc="batches",
            disable=False,
            total=self.config.num_updates,
        ):
            batch = next(train_iter)

            if "info" in batch:
                del batch["info"]
            batch = Batch(**batch)
            metrics, extra = self.model.update(next(self.rng_seq), batch)

            # logging.info(f"step: {step}, metrics: {metrics}")
            if self.wandb_run is not None:
                metrics = gutl.prefix_dict_keys(metrics, prefix="train/")
                self.wandb_run.log(metrics)

            if ((train_step + 1) % self.eval_every) == 0:
                eval_metrics = self.eval(step=train_step + 1)

                if ((train_step + 1) % self.eval_every) == 0:
                    for eval_env_id, env_eval_metrics in eval_metrics.items():
                        eval_metrics[eval_env_id] = gutl.prefix_dict_keys(
                            env_eval_metrics, prefix=f"{eval_env_id}/eval/"
                        )

                if self.wandb_run is not None:
                    self.wandb_run.log(eval_metrics)

            if ((train_step + 1) % self.config.log_interval) == 0:
                print(f"step: {train_step}, metrics: {metrics}")

                # visualize some of the next observation predictions
                if self.wandb_run is not None and "next_obs_pred" in extra:
                    self.visualize("train", batch, extra, self.config.env.env_id)

        self.eval(step=self.config.num_updates)

    def visualize(self, stage, batch, extra, env_id):
        # visualize some of the next observation predictions
        if "next_obs_pred" in extra:
            # if we're using ViT, this is [B, T+1, H, W, C], else this is [B, H, W, C]
            next_obs_pred = extra["next_obs_pred"]

            if self.config.model.use_vit:
                num_ex = 4
                gt_next_obs = batch.observations[:num_ex, 1:]
            else:
                num_ex = 16
                gt_next_obs = batch.observations[:num_ex, -1]

            next_obs_pred = next_obs_pred[0][:num_ex]
            next_obs_pred = np.array(next_obs_pred)

            # also record the difference between the ground truth and the prediction
            # as a binary array (0 if the pixel is the same, 1 if it is different)
            # diff = np.where(gt_next_obs != next_obs_pred, 1, 0)
            diff = np.abs(gt_next_obs - next_obs_pred)

            # stack and interleave ground truth and prediction
            to_show = np.stack([gt_next_obs, next_obs_pred, diff], axis=1)
            to_show = to_show.reshape(-1, *to_show.shape[2:])

            if to_show.ndim == 5:
                # flatten first two dimensions
                to_show = einops.rearrange(to_show, "b t h w c -> (b t) h w c")

            to_show = [custom_to_pil(x) for x in to_show]

            if self.config.model.use_vit:
                to_show = make_image_grid(to_show, num_rows=num_ex * 3)
            else:
                to_show = make_image_grid(to_show, num_rows=4)

            if self.wandb_run:
                self.wandb_run.log(
                    {f"{stage}/{env_id}/next_obs_pred": wandb.Image(to_show)}
                )

    def eval(self, step: int):
        logging.info("running evaluation")
        metrics = {}
        for eval_env_id in self.eval_dataloaders.keys():
            eval_metrics = self.eval_env(step, eval_env_id)
            metrics[eval_env_id] = eval_metrics

            # save model
            if self.config.mode == "train" and self.config.env.env_id == eval_env_id:
                self.save_model(self.model.save_dict, eval_metrics, step)

            # write to log file
            log_eval_metrics = dict(
                jtu.tree_map(lambda x: np.round(float(x), 2), eval_metrics)
            )

            with open(self.log_dir / f"eval_{eval_env_id}.txt", "a+") as f:
                f.write(f"{step}, {log_eval_metrics}\n")

            logging.info(f"eval [{eval_env_id}]: {log_eval_metrics}")

        return metrics

    def eval_env(self, step: int, eval_env_id: str):
        logging.info(f"running evaluation for {eval_env_id}")
        eval_metrics = dd(list)

        # run on eval batches
        if self.eval_dataloaders is not None:
            # for _ in range(self.num_eval_batches):
            #     batch = next(self.eval_dataloader)

            eval_dataloader = self.eval_dataloaders[eval_env_id]
            eval_iter = eval_dataloader.repeat().as_numpy_iterator()
            # data_loading_time = time.time()

            for eval_step in tqdm.tqdm(range(50), desc="eval batches"):
                batch = next(eval_iter)
                # logging.info(f"data loading time: {time.time() - data_loading_time}")
                if "info" in batch:
                    del batch["info"]

                batch = Batch(**batch)
                update_time = time.time()
                metrics, extra = self.model.update(
                    next(self.rng_seq), batch, update_model=False
                )

                # logging.info(f"update time: {time.time() - update_time}")
                for k, v in metrics.items():
                    # make sure it is scalar
                    if not v.ndim == 0:
                        continue  # skip non-scalar metrics
                    eval_metrics[k].append(v)

                # data_loading_time = time.time()

                if eval_step == 0:
                    # visualize some of the next observation predictions
                    if "next_obs_pred" in extra:
                        self.visualize("eval", batch, extra, eval_env_id)

            for k, v in eval_metrics.items():
                eval_metrics[k] = jnp.mean(jnp.array(v))

        # run rollouts
        rollout_start = time.time()
        if hasattr(self.model, "get_action") and self.config.run_eval_rollouts:
            if self.config.env.env_name == "atari":
                rollout_fn = run_rollouts_atari
            elif self.config.env.env_name == "procgen":
                rollout_fn = run_rollouts_procgen
            else:
                rollout_fn = run_rollouts

            rollout_metrics, *_ = rollout_fn(
                rng=next(self.rng_seq),
                agent=self.model,
                env=self.eval_envs[eval_env_id][0],
                env_id=eval_env_id,
                config=self.config,
                action_dim=self.model.input_action_dim,
                # steps_per_rollout=self.steps_per_rollout,
                wandb_run=self.wandb_run,
                # prompts=(
                #     self.prompts[eval_env_id] if eval_env_id in self.prompts else None
                # ),
            )
            eval_metrics.update(rollout_metrics)

        eval_metrics["time/rollouts"] = time.time() - rollout_start
        return eval_metrics
