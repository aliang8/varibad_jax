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
from functools import partial
from collections import defaultdict as dd
from varibad_jax.utils.rollout import run_rollouts, run_rollouts_procgen

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
        self.train_dataset, self.eval_dataset, self.prompt_dataset = load_data(
            config=config, rng=next(self.rng_seq)
        )
        logging.info(f"data loading time: {time.time() - data_load_time}")

        # these are trajectories
        self.prompts = self.sample_prompts()

        # # add labels
        # if "vpt_bc" in self.config.model.name or "lam" in self.config.model.name:
        #     # edit action labels

        #     # first load pretrained model
        #     # load pretrained IDM for predicting the latent actions from observations
        #     ckpt_path = self.config.model.vpt_idm_ckpt

        #     if self.config.model.idm_nt != -1:
        #         ckpt_path = re.sub(
        #             r"nt-\d+", f"nt-{self.config.model.idm_nt}", ckpt_path
        #         )

        #     config = Path(ckpt_path) / "config.json"
        #     with open(config, "r") as f:
        #         vpt_cfg = ConfigDict(json.load(f))

        #     key, vpt_key = jax.random.split(next(self.rng_seq), 2)

        #     vpt = VPT(
        #         config=vpt_cfg.model,
        #         key=vpt_key,
        #         load_from_ckpt=True,
        #         ckpt_dir=Path(ckpt_path),
        #         observation_shape=(1, 1),  # doesn't matter
        #         action_dim=self.action_dim,
        #         input_action_dim=self.input_action_dim,
        #         continuous_actions=self.continuous_actions,
        #         task_dim=self.config.env.task_dim,
        #     )

        #     # predict the action from observations with pretrained model
        #     vpt_action_pred, _ = vpt.model.apply(
        #         vpt._ts.params,
        #         vpt._ts.state,
        #         vpt_key,
        #         vpt,
        #         states=batch.observations.astype(jnp.float32),
        #         is_training=False,
        #     )
        #     import ipdb

        #     ipdb.set_trace()

        self.eval_dataloaders = {}

        if self.config.env.env_name == "procgen":
            self.train_dataloader = self.train_dataset.get_iter(
                self.config.data.batch_size
            )
            for eval_env_id in self.eval_dataset.keys():
                self.eval_dataloaders[eval_env_id] = self.eval_dataset.get_iter(
                    self.config.data.batch_size
                )

            # number of batches ot iterate over each epoch of training
            self.num_train_batches = 50
            self.num_eval_batches = 2
            obs_shape = (64, 64, 3)
        else:
            self.train_dataloader = create_data_loader(
                self.train_dataset, data_cfg=config.data
            )
            logging.info(f"Number of train batches: {len(self.train_dataloader)}")
            for eval_env_id, eval_dataset in self.eval_dataset.items():
                if len(eval_dataset["observations"]) > 0:
                    eval_dataloader = create_data_loader(
                        eval_dataset, data_cfg=config.data
                    )
                    logging.info(f"Number of eval batches: {len(eval_dataloader)}")
                    self.eval_dataloaders[eval_env_id] = eval_dataloader

            if config.data.data_type == "trajectories":
                obs_shape = self.train_dataset["observations"].shape[2:]
            else:
                obs_shape = self.train_dataset["observations"].shape[1:]

        # print batch item shapes
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
        )

        if self.config.num_evals != -1:
            self.eval_every = int(self.config.num_epochs // self.config.num_evals)
        elif self.config.eval_perc != -1:
            self.eval_every = int(self.config.num_epochs * self.config.eval_perc)
        else:
            self.eval_every = self.config.eval_interval

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
            eval_metrics = self.eval(epoch=0)

        # train
        for epoch in tqdm.tqdm(range(self.config.num_epochs), desc="epochs"):
            # iterate over batches of data
            start_time = time.time()
            train_ep_metrics = dd(list)
            # for _ in range(self.num_train_batches):
            #     batch = next(self.train_dataloader)
            for batch in self.train_dataloader:
                if "info" in batch:
                    del batch["info"]
                batch = Batch(**batch)
                metrics = self.model.update(next(self.rng_seq), batch)
                for k, v in metrics.items():
                    train_ep_metrics[k].append(v)

            for k, v in train_ep_metrics.items():
                train_ep_metrics[k] = jnp.mean(jnp.array(v))

            epoch_time = time.time() - start_time
            train_ep_metrics["time/epoch"] = epoch_time
            train_ep_metrics["misc/lr"] = self.model._ts.opt_state.hyperparams["lr"]

            if ((epoch + 1) % self.eval_every) == 0:
                print("train: ", train_ep_metrics)
                eval_metrics = self.eval(epoch=epoch + 1)

                if self.config.data.resample_prompts_every_eval:
                    logging.info("getting new set of demo eval prompts")
                    # get new set of prompts for evaluation
                    self.prompts = self.sample_prompts()

            if self.wandb_run is not None:
                metrics = gutl.prefix_dict_keys(train_ep_metrics, prefix="train/")

                if ((epoch + 1) % self.eval_every) == 0:
                    for eval_env_id, env_eval_metrics in eval_metrics.items():
                        prefixed_eval_metrics = gutl.prefix_dict_keys(
                            env_eval_metrics, prefix=f"{eval_env_id}/eval/"
                        )
                        metrics.update(prefixed_eval_metrics)

                self.wandb_run.log(metrics)

        self.eval(epoch=self.config.num_epochs)

    def eval(self, epoch: int):
        metrics = {}
        for eval_env_id in self.eval_dataloaders.keys():
            eval_metrics = self.eval_env(epoch, eval_env_id)
            metrics[eval_env_id] = eval_metrics

            # save model
            if self.config.mode == "train" and self.config.env.env_id == eval_env_id:
                self.save_model(self.model.save_dict, eval_metrics, epoch)

            # write to log file
            log_eval_metrics = dict(
                jtu.tree_map(lambda x: np.round(float(x), 2), eval_metrics)
            )

            with open(self.log_dir / f"eval_{eval_env_id}.txt", "a+") as f:
                f.write(f"{epoch}, {log_eval_metrics}\n")

            logging.info(f"eval [{eval_env_id}]: {log_eval_metrics}")

        return metrics

    def eval_env(self, epoch: int, eval_env_id: str):
        eval_metrics = dd(list)

        # run on eval batches
        if self.eval_dataloaders is not None:
            # for _ in range(self.num_eval_batches):
            #     batch = next(self.eval_dataloader)

            eval_dataloader = self.eval_dataloaders[eval_env_id]
            for batch in eval_dataloader:
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

        # run rollouts
        if hasattr(self.model, "get_action") and self.config.run_eval_rollouts:
            if self.config.env.env_name == "procgen":
                rollout_metrics, *_ = run_rollouts_procgen(
                    rng=next(self.rng_seq),
                    agent=self.model,
                    env=self.eval_envs,
                    config=self.config,
                    wandb_run=self.wandb_run,
                )
            else:
                rollout_metrics, *_ = run_rollouts(
                    rng=next(self.rng_seq),
                    agent=self.model,
                    env=self.eval_envs[eval_env_id][0],
                    env_id=eval_env_id,
                    config=self.config,
                    action_dim=self.model.input_action_dim,
                    steps_per_rollout=self.steps_per_rollout,
                    wandb_run=self.wandb_run,
                    prompts=(
                        self.prompts[eval_env_id]
                        if eval_env_id in self.prompts
                        else None
                    ),
                )
            eval_metrics.update(rollout_metrics)
        return eval_metrics
