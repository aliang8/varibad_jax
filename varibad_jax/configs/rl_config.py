"""Config for posterior transformer training."""

from flax import struct
from ml_collections import config_dict
import numpy as np
from varibad_jax.configs.base_config import get_config as get_base_config


def get_config(config_string: str = None):
    """Defines base config for training."""

    # =============================================================
    # Environment config
    # =============================================================
    config = get_base_config()

    config.trainer = "rl_trainer"
    config.policy.pass_latent_to_policy = False
    config.policy.pass_task_to_policy = True

    return config
