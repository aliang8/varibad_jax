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

        image_encoder_config.embedding_dim = 8  # TODO: fix this
    else:
        image_encoder_config = None

    config.trainer = "rl"
    # =============================================================
    # Policy configs
    # =============================================================
    policy_config = config_dict.ConfigDict(
        dict(
            image_encoder_config=image_encoder_config,
            image_obs=config.env.get_ref("image_obs"),
            task_dim=config.env.get_ref("task_dim"),
            pass_state_to_policy=True,
            pass_latent_to_policy=False,
            pass_belief_to_policy=False,
            pass_task_to_policy=True,
            use_hyperx_bonuses=False,
            mlp_layers=[32, 32],
            actor_activation_function="tanh",
            algo="ppo",
            name="ppo",
            optimizer="adam",
            num_epochs=2,
            num_minibatch=4,
            clip_eps=0.05,
            lr=7e-4,
            eps=1e-8,
            value_loss_coeff=0.5,
            entropy_coeff=0.01,
            gamma=0.95,
            use_gae=True,
            tau=0.95,
            max_grad_norm=0.5,
            embedding_dim=16,
            anneal_lr=False,
        )
    )
    config.policy = policy_config

    config.notes = "RL"
    config.tags = ["rl", "jax"]
    config.keys_to_include = {
        "trainer": None,
        "env": ["env_name"],
        "policy": ["algo", "pass_latent_to_policy"],
    }

    return config
