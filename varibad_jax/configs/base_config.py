"""Config for posterior transformer training."""

from flax import struct
from ml_collections.config_dict import ConfigDict
import numpy as np


def get_config(config_string: str = None):
    """Defines base config for training."""

    # =============================================================
    # Environment config
    # =============================================================
    config = ConfigDict()

    env_config = ConfigDict()

    # define env specific configs
    envs = {
        "gridworld": ConfigDict(
            dict(
                task_dim=2,
                env_name="gridworld",
                env_id="GridNaviJAX-v0",
                num_episodes_per_rollout=4,
                steps_per_rollout=15,
                num_processes=16,
                normalize_rews=False,
                image_obs=False,
                env_kwargs=dict(
                    grid_size=5, max_episode_steps=15, num_actions=5, random_init=True
                ),
            )
        ),
        "xland": ConfigDict(
            dict(
                env_name="xland",
                env_kwargs=dict(),
                benchmark_path="/scr/aliang80/varibad_jax/varibad_jax/envs/xland_benchmarks/test_ruleset.pkl",
                # preloaded_benchmark="small-1m",
                ruleset_id=-1,
                env_id="XLand-MiniGridCustom-R1-7x7",
                num_episodes_per_rollout=1,
                steps_per_rollout=30,
                num_processes=16,
                normalize_rews=False,
                image_obs=True,
                symbolic_obs=True,
                task_dim=5,  # TODO: tbd fix this
                full_observability=False,
            )
        ),
        "procgen": ConfigDict(
            dict(
                env_name="procgen",
                env_id="bigfish",
                num_envs=2,
                image_obs=True,
                gamma=0.999,
                num_episodes_per_rollout=1,
                task_dim=1,
                steps_per_rollout=200,
            )
        ),
        "minatar": ConfigDict(
            dict(
                env_name="minatar",
                env_id="Breakout-MinAtar",
                num_processes=16,
                image_obs=True,
                num_episodes_per_rollout=1,
                task_dim=1,
                steps_per_rollout=1000,
            )
        ),
        "atari": ConfigDict(
            dict(
                env_name="atari",
                env_id="Breakout",
                num_envs=16,
                image_obs=True,
                num_episodes_per_rollout=1,
                task_dim=1,
                steps_per_rollout=100,
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

    # number of updates
    config.log_level = "info"
    config.enable_jit = True
    config.log_interval = 500

    # saving checkpoints
    config.save_interval = 100
    config.save_key = "episode_return"
    config.best_metric = "max"

    config.eval_interval = -1
    config.num_evals = -1
    config.eval_perc = -1
    config.disable_tqdm = False
    config.smoke_test = True

    # number of warmup steps before training
    config.warmup_steps = 10000
    # where to save experiment artifacts (videos, checkpoints, etc)
    config.root_dir = "/scr/aliang80/varibad_jax/varibad_jax/"
    config.exp_dir = "/scr/aliang80/varibad_jax/varibad_jax/"
    config.exp_name = ""
    config.group_name = ""
    config.ray_logdir = "ray_results"
    config.overwrite = True
    config.wandb_entity = "glamor"
    config.wandb_project = "varibad_jax"

    # for rollout visualization
    config.fps = 5
    config.num_eval_rollouts = 50

    # number of eval rollout videos to save
    config.visualize_rollouts = True
    config.num_eval_rollouts_render = 10
    config.skip_first_eval = False
    config.run_eval_rollouts = True

    # for saving checkpoints
    config.max_ckpt_to_keep = None

    # resume training / evaluation
    config.load_from_ckpt = False
    config.ckpt_step = -1
    config.model_ckpt_dir = ""

    # ray
    config.cpu = 5
    config.gpu = 0.1

    config.num_xla_devices = 1
    return config
