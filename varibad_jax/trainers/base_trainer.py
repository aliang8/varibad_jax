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
from ml_collections import FrozenConfigDict
from pathlib import Path
from varibad_jax.envs.utils import make_envs
from varibad_jax.envs.xland import make_envs as make_envs_xland
import gymnasium as gym
from jax import config as jax_config


class BaseTrainer:
    def __init__(self, config: FrozenConfigDict):
        self.config = config
        self.global_step = 0
        self.rng_seq = hk.PRNGSequence(config.seed)

        if self.config.use_wb:
            self.wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                entity="glamor",
                project="varibad_jax",
                name=self.config.exp_name,
                notes=self.config.notes,
                tags=self.config.tags,
                # track hyperparameters and run metadata
                config=self.config,
            )
        else:
            self.wandb_run = None

        # setup log dirs
        self.exp_dir = Path(self.config.root_dir) / self.config.exp_name
        print("experiment dir: ", self.exp_dir)

        self.ckpt_dir = self.exp_dir / "model_ckpts"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir = self.exp_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)

        # create env
        if self.config.env == "gridworld":
            self.envs = make_envs(
                self.config.env.env_id,
                seed=self.config.seed,
                num_envs=self.config.env.num_processes,
                num_episodes_per_rollout=self.config.env.num_episodes_per_rollout,
            )
        else:

            self.envs = make_envs_xland(
                env_id=self.config.env.env_id,
                seed=self.config.seed,
                num_envs=self.config.env.num_processes,
                num_episodes_per_rollout=self.config.env.num_episodes_per_rollout,
                benchmark_path=self.config.env.benchmark_path,
                ruleset_id=self.config.env.ruleset_id,
            )

        self.obs_shape = self.envs.observation_space.shape
        if isinstance(self.envs.action_space, gym.spaces.Discrete):
            self.action_dim = self.envs.action_space.n
        else:
            self.action_dim = self.envs.action_space.shape[0]

        if self.config.log_level == "info":
            logging.set_verbosity(logging.INFO)
        elif self.config.log_level == "debug":
            logging.set_verbosity(logging.DEBUG)

        if not self.config.enable_jit:
            jax_config.update("jax_disable_jit", True)

        logging.info(f"obs_shape: {self.obs_shape}, action_dim: {self.action_dim}")

    def create_ts(self):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def test(self, epoch):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
