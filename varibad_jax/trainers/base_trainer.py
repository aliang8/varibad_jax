from absl import app, logging
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import os
import tqdm
import pickle
import time
import wandb
import optax
import einops
import json
import shutil
import jax.tree_util as jtu
from ml_collections import FrozenConfigDict
from pathlib import Path
from varibad_jax.envs.utils import make_envs
import gymnasium as gym
from jax import config as jax_config
from varibad_jax.utils.rollout import (
    eval_rollout_with_belief_model,
    eval_rollout,
    eval_rollout_dt,
)


class BaseTrainer:
    def __init__(self, config: FrozenConfigDict):
        self.config = config

        print(self.config)

        self.global_step = 0
        self.rng_seq = hk.PRNGSequence(config.seed)
        np.random.seed(config.seed)

        # setup log dirs
        self.exp_dir = Path(self.config.exp_dir)
        print("experiment dir: ", self.exp_dir)

        if self.exp_dir.exists() and not self.config.overwrite:
            logging.info(
                f"experiment dir {self.exp_dir} already exists, will create a slightly different one"
            )
            # raise ValueError("experiment dir already exists")
            rand_str = str(int(time.time()))
            self.exp_dir = self.exp_dir.parent / self.exp_dir.name / rand_str
            logging.info(f"new experiment dir: {self.exp_dir}")
        else:
            logging.info(f"overwriting existing experiment dir {self.exp_dir}")
            shutil.rmtree(self.exp_dir, ignore_errors=True)
            self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = self.exp_dir / "model_ckpts"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir = self.exp_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)

        # save config to json file readable
        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=4)

        if self.config.use_wb:
            self.wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                entity="glamor",
                project="varibad_jax",
                name=self.config.exp_name,
                notes=self.config.notes,
                tags=self.config.tags,
                # track hyperparameters and run metadata
                config=self.config.to_dict(),
                group=self.config.group_name,
            )
        else:
            self.wandb_run = None

        # create env
        self.envs, self.env_params = make_envs(**self.config.env)
        self.eval_envs, _ = make_envs(training=False, **self.config.env)

        self.jit_reset = jax.vmap(jax.jit(self.envs.reset), in_axes=(None, 0))
        self.jit_step = jax.vmap(jax.jit(self.envs.step), in_axes=(None, 0, 0))

        self.obs_shape = self.envs.observation_space.shape
        self.continuous_actions = not isinstance(
            self.envs.action_space, gym.spaces.Discrete
        )
        if isinstance(self.envs.action_space, gym.spaces.Discrete):
            self.action_dim = self.envs.action_space.n
            self.input_action_dim = 1
        else:
            self.input_action_dim, self.action_dim = self.envs.action_space.shape[0]

        # this is for the case with fixed length sessions
        self.steps_per_rollout = (
            config.env.num_episodes_per_rollout * self.envs.max_episode_steps
        )

        if self.config.log_level == "info":
            logging.set_verbosity(logging.INFO)
        elif self.config.log_level == "debug":
            logging.set_verbosity(logging.DEBUG)

        if not self.config.enable_jit:
            jax_config.update("jax_disable_jit", True)

        logging.info(f"obs_shape: {self.obs_shape}, action_dim: {self.action_dim}")
        logging.info(f"env params: {self.env_params}")

        if config.best_metric == "max":
            self.best_metric = float("-inf")
        else:
            self.best_metric = float("inf")

    def create_ts(self):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def test(self, epoch):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def save_model(self, ckpt_dict, metrics, iter_idx: int = None):
        if self.config.save_key and self.config.save_key in metrics:
            # import ipdb; ipdb.set_trace()
            key = self.config.save_key
            if (
                self.config.best_metric == "max" and metrics[key] > self.best_metric
            ) or (self.config.best_metric == "min" and metrics[key] < self.best_metric):
                self.best_metric = metrics[key]
                ckpt_file = self.ckpt_dir / f"best.pkl"
                logging.info(
                    f"new best value: {metrics[key]}, saving best model at epoch {iter_idx + 1} to {ckpt_file}"
                )
                with open(ckpt_file, "wb") as f:
                    pickle.dump(ckpt_dict, f)

                # create a file with the best metric in the name, use a placeholder
                best_ckpt_file = self.ckpt_dir / "best.txt"
                with open(best_ckpt_file, "w") as f:
                    f.write(f"{iter_idx + 1}, {metrics[key]}")

        # also save model to ckpt everytime we run evaluation
        ckpt_file = Path(self.ckpt_dir) / f"ckpt_{iter_idx + 1}.pkl"
        logging.debug(f"saving checkpoint to {ckpt_file}")
        with open(ckpt_file, "wb") as f:
            pickle.dump(ckpt_dict, f)
