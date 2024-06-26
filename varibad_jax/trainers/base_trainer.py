from absl import app, logging
import copy
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import os
import tqdm
import pickle
import time
import torch
import wandb
import optax
import einops
import json
import gymnax
import shutil
import jax.tree_util as jtu
import tensorflow as tf
from ml_collections import FrozenConfigDict
from pathlib import Path
from varibad_jax.envs.utils import make_envs, make_procgen_envs, make_atari_envs
import gymnasium as gym
from jax import config as jax_config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress debug warning messages


class BaseTrainer:
    def __init__(self, config: FrozenConfigDict):
        self.config = config

        print(self.config)

        self.global_step = 0
        self.rng_seq = hk.PRNGSequence(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        tf.random.set_seed(config.seed)

        # setup log dirs
        # split exp_dir by ,
        root_dir = Path(config.exp_dir).parent
        base_name = Path(config.exp_dir).name
        subfolders = base_name.split(",")

        self.exp_dir = Path(root_dir, *subfolders)
        print("experiment dir: ", self.exp_dir)
        self.ckpt_dir = self.exp_dir / "model_ckpts"
        self.video_dir = self.exp_dir / "videos"
        self.wandb_run = None

        self.num_devices = jax.device_count()
        self.num_local_devices = jax.local_device_count()

        logging.info(
            f"num_devices: {self.num_devices}, num_local_devices: {self.num_local_devices}"
        )

        if self.config.mode == "train":
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

            self.log_dir = self.exp_dir / "logs"
            self.ckpt_dir = self.exp_dir / "model_ckpts"
            self.video_dir = self.exp_dir / "videos"
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.video_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # save config to json file readable
            with open(self.exp_dir / "config.json", "w") as f:
                json.dump(self.config.to_dict(), f, indent=4)

            if self.config.use_wb:
                self.wandb_run = wandb.init(
                    # set the wandb project where this run will be logged
                    entity=config.wandb_entity,
                    project=config.wandb_project,
                    name=self.config.exp_name,
                    notes=self.config.notes,
                    tags=self.config.tags,
                    # track hyperparameters and run metadata
                    config=self.config.to_dict(),
                    group=self.config.group_name,
                )

        # create env
        logging.info("creating environments...")
        if self.config.env.env_name == "procgen":
            # not jax jit compatible
            self.envs = make_procgen_envs(**self.config.env)
            self.config.env.num_envs = self.config.num_eval_rollouts
            self.eval_envs = {
                "bigfish": (make_procgen_envs(training=False, **self.config.env), {})
            }
            self.obs_shape = self.envs.single_observation_space.shape
            self.continuous_actions = False
            self.input_action_dim = 1
            self.action_dim = self.envs.single_action_space.n
        elif self.config.env.env_name == "atari":
            self.envs = make_atari_envs(**self.config.env)
            self.config.env.num_envs = self.config.num_eval_rollouts
            self.eval_envs = {}
            for env_id in self.config.env.eval_env_ids:
                self.eval_envs[env_id] = (
                    make_atari_envs(training=False, **self.config.env),
                    {},
                )
            # self.obs_shape = self.envs.observation_spec().shape
            self.obs_shape = self.envs.single_observation_space.shape
            self.continuous_actions = False
            self.input_action_dim = 1
            self.action_dim = self.envs.single_action_space.n
            # self.action_dim = self.envs.action_spec().num_values
            self.steps_per_rollout = self.config.env.steps_per_rollout
        elif self.config.env.env_name == "xland":
            self.envs, self.env_params = make_envs(**self.config.env)
            logging.info(f"env [{self.config.env.env_id}] params: {self.env_params}")

            # create envs for each scenario we want to evaluate on
            self.eval_envs = {}
            for env_id in self.config.env.eval_env_ids:
                env_cfg = copy.deepcopy(self.config.env)
                env_cfg.env_id = env_id
                eval_envs, eval_env_params = make_envs(training=False, **env_cfg)
                self.eval_envs[env_id] = (eval_envs, eval_env_params)
                logging.info(f"eval env [{env_id}] params: {eval_env_params}")

            self.jit_reset = jax.vmap(jax.jit(self.envs.reset), in_axes=(None, 0))
            self.jit_step = jax.vmap(jax.jit(self.envs.step), in_axes=(None, 0, 0))

            self.obs_shape = self.envs.observation_space.shape
            self.continuous_actions = not isinstance(
                self.envs.action_space, gym.spaces.Discrete
            ) and not isinstance(
                self.envs.action_space, gymnax.environments.spaces.Discrete
            )

            if not self.continuous_actions:
                self.action_dim = self.envs.action_space.n
                self.input_action_dim = 1
            else:
                self.input_action_dim, self.action_dim = self.envs.action_space.shape[0]

            # this is for the case with fixed length sessions
            self.steps_per_rollout = (
                config.env.num_episodes_per_rollout * self.envs.max_episode_steps
            )

            try:
                self.task_dim = self.env_params.task_dim
            except:
                self.task_dim = self.config.env.task_dim

            logging.info(f"task_dim: {self.task_dim}")
        else:
            raise ValueError(f"env_name {self.config.env.env_name} not supported")

        if self.config.log_level == "info":
            logging.set_verbosity(logging.INFO)
        elif self.config.log_level == "debug":
            logging.set_verbosity(logging.DEBUG)

        if not self.config.enable_jit:
            jax_config.update("jax_disable_jit", True)

        logging.info(f"obs_shape: {self.obs_shape}, action_dim: {self.action_dim}")

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

    def save_model(self, ckpt_dict, metrics, iter: int = None):
        if self.config.save_key and self.config.save_key in metrics:
            # import ipdb; ipdb.set_trace()
            key = self.config.save_key
            if (
                self.config.best_metric == "max" and metrics[key] > self.best_metric
            ) or (self.config.best_metric == "min" and metrics[key] < self.best_metric):
                self.best_metric = metrics[key]
                ckpt_file = self.ckpt_dir / f"best.pkl"
                logging.info(
                    f"new best value: {metrics[key]}, saving best model at epoch {iter} to {ckpt_file}"
                )
                with open(ckpt_file, "wb") as f:
                    pickle.dump(ckpt_dict, f)

                # create a file with the best metric in the name, use a placeholder
                best_ckpt_file = self.ckpt_dir / "best.txt"
                with open(best_ckpt_file, "w") as f:
                    f.write(f"{iter}, {metrics[key]}")

        # also save model to ckpt everytime we run evaluation
        ckpt_file = Path(self.ckpt_dir) / f"ckpt_{iter:04d}.pkl"
        logging.info(f"saving checkpoint to {ckpt_file}")
        with open(ckpt_file, "wb") as f:
            pickle.dump(ckpt_dict, f)
