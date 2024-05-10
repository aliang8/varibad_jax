"""Config for posterior transformer training."""

from flax import struct
from ml_collections import config_dict
import numpy as np
from varibad_jax.configs.base_config import get_config as get_base_config
from varibad_jax.configs.model_configs import image_encoder_configs


def get_config(config_string: str = None):
    """Defines base config for training."""

    # =============================================================
    # Environment config
    # =============================================================
    config = get_base_config(config_string)

    if "xland" in config_string:
        image_encoder_config = None
        for k, v in image_encoder_configs.items():
            if k in config_string:
                image_encoder_config = v
                break
        if image_encoder_config is None:
            raise ValueError(
                "No image encoder config found for the given config string"
            )

        image_encoder_config.embedding_dim = 64  # TODO: fix this
    else:
        image_encoder_config = None

    config.trainer = "rl"

    policy_config = config_dict.ConfigDict(
        dict(
            pass_state_to_policy=True,
            pass_latent_to_policy=False,
            pass_belief_to_policy=False,
            pass_task_to_policy=False,
            mlp_layer_sizes=[128, 128],
            use_rnn_policy=False,
            rnn_hidden_size=256,
            gaussian_policy=False,
            embedding_dim=128,
            image_obs=config.env.get_ref("image_obs"),
            image_encoder_config=image_encoder_config,
        )
    )

    # =============================================================
    # RL algorithm configs
    # =============================================================
    algo_config = config_dict.ConfigDict(
        dict(
            policy=policy_config,
            image_obs=config.env.get_ref("image_obs"),
            task_dim=config.env.get_ref("task_dim"),
            use_hyperx_bonuses=False,
            name="ppo",
            num_epochs=1,
            num_minibatches=8,
            clip_eps=0.2,
            lr=7e-4,
            eps=1e-8,
            value_loss_coeff=0.5,
            entropy_coeff=0.01,
            gamma=0.99,
            use_gae=True,
            tau=0.95,
            max_grad_norm=0.5,
            use_lr_scheduler=False,
        )
    )
    config.model = algo_config

    config.notes = "RL"
    config.tags = ["rl", "jax"]
    config.keys_to_include = {
        "trainer": None,
        "env": ["env_name"],
    }

    # RL training config
    config.num_envs = 1024
    config.num_steps = 16
    return config
