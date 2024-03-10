from absl import app
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
import gymnasium as gym


class BaseTrainer:
    def __init__(self, config: FrozenConfigDict):
        self.config = config
        self.global_step = 0
        self.rng_seq = hk.PRNGSequence(config.seed)

        if self.config.use_wb:
            self.wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                entity="glamor",
                project="data_augmentation",
                name=self.config.exp_name,
                notes=self.config.notes,
                tags=self.config.tags,
                # track hyperparameters and run metadata
                config=self.config,
            )
        else:
            self.wandb_run = None

        # setup log dirs
        self.exp_dir = Path(self.config.root_dir) / "log" / self.config.exp_name
        print("experiment dir: ", self.exp_dir)

        self.ckpt_dir = self.exp_dir / "model_ckpts"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir = self.exp_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)

        # create env
        self.envs = make_envs(
            self.config.env.env_id,
            seed=self.config.seed,
            num_envs=self.config.env.num_processes,
        )

        self.state_dim = self.envs.observation_space.shape[0]
        if isinstance(self.envs.action_space, gym.spaces.Discrete):
            self.action_dim = self.envs.action_space.n
        else:
            self.action_dim = self.envs.action_space.shape[0]

    def create_ts(self):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def test(self, epoch):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
