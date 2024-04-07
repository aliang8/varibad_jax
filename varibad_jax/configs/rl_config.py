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
    config = get_base_config(config_string)

    config.trainer = "rl"
    # =============================================================
    # Policy configs
    # =============================================================
    policy_config = config_dict.ConfigDict()
    # policy_config.image_encoder_config = image_encoder_config
    policy_config.image_obs = config.env.get_ref("image_obs")
    policy_config.pass_state_to_policy = True
    policy_config.pass_latent_to_policy = False
    policy_config.pass_belief_to_policy = False
    policy_config.pass_task_to_policy = True
    policy_config.mlp_layers = [32, 32]
    policy_config.actor_activation_function = "tanh"
    policy_config.algo = "ppo"
    policy_config.optimizer = "adam"
    policy_config.num_epochs = 2
    policy_config.num_minibatch = 4
    policy_config.clip_eps = 0.05
    policy_config.lr = 7e-4
    policy_config.eps = 1e-8
    policy_config.value_loss_coeff = 0.5
    policy_config.entropy_coeff = 0.01
    policy_config.gamma = 0.95
    policy_config.use_gae = True
    policy_config.tau = 0.95
    policy_config.max_grad_norm = 0.5
    policy_config.embedding_dim = 16
    config.policy = policy_config

    config.notes = "RL"
    config.tags = ["rl", "jax"]
    config.keys_to_include = {
        "trainer": None,
        "env": ["env_name"],
        "policy": ["algo", "pass_latent_to_policy"],
    }

    return config
