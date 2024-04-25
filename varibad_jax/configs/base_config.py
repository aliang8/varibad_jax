"""Config for posterior transformer training."""

from flax import struct
from ml_collections import config_dict
import numpy as np


def get_config(config_string: str = None):
    """Defines base config for training."""

    # =============================================================
    # Environment config
    # =============================================================
    config = config_dict.ConfigDict()

    env_config = config_dict.ConfigDict()

    # define env specific configs
    envs = {
        "gridworld": config_dict.ConfigDict(
            dict(
                env_name="gridworld",
                env_id="GridNaviJAX-v0",
                num_episodes_per_rollout=4,
                steps_per_rollout=15,
                num_processes=16,
                normalize_rews=False,
                image_obs=False,
                env_kwargs=dict(grid_size=5, max_episode_steps=15, num_actions=5),
            )
        ),
        "xland": config_dict.ConfigDict(
            dict(
                env_name="xland",
                env_kwargs=dict(view_size=5, height=9, width=9),
                benchmark_path="/home/anthony/varibad_jax/varibad_jax/envs/xland_benchmarks/test_ruleset.pkl",
                ruleset_id=0,
                env_id="XLand-MiniGrid-R1-9x9",
                num_episodes_per_rollout=4,
                steps_per_rollout=20,
                num_processes=16,
                normalize_rews=False,
                image_obs=True,
            )
        ),
    }

    for k, v in envs.items():
        if k in config_string:
            env_config.update(v)

    config.env = env_config

    # =============================================================
    # Experiment stuff
    # =============================================================
    config.num_frames = 20_000_000
    config.trainer = "vae"
    config.seed = 521
    config.mode = "train"
    config.use_wb = False
    config.lr_anneal_method = "warmup_exp_decay"

    # number of updates
    config.log_level = "info"
    config.enable_jit = True
    config.log_interval = 10

    # saving checkpoints
    config.save_interval = 100
    config.save_key = "episode_return"
    config.best_metric = "max"

    config.eval_interval = 100
    config.disable_tqdm = True
    config.smoke_test = True

    # number of warmup steps before training
    config.warmup_steps = 10000
    # where to save experiment artifacts (videos, checkpoints, etc)
    config.root_dir = "/home/anthony/varibad_jax/varibad_jax/"
    config.exp_dir = "/home/anthony/varibad_jax/varibad_jax/"
    config.exp_name = "vb_jax"
    config.group_name = ""
    config.ray_logdir = "ray_results"

    # for rollout visualization
    config.fps = 5
    config.num_eval_rollouts = 10

    # number of eval rollout videos to save
    config.visualize_rollouts = False
    config.num_eval_rollouts_save = 3
    config.skip_first_eval = False

    # for saving checkpoints
    config.max_ckpt_to_keep = None

    # resume training / evaluation
    config.load_from_ckpt = False
    config.ckpt_step = -1
    config.model_ckpt_dir = ""

    # ray
    config.cpu = 5
    config.gpu = 0.1
    return config
