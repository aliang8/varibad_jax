from varibad_jax.configs.base_config import get_config as get_base_config
from varibad_jax.configs.model_configs import (
    transformer_config,
    image_encoder_config,
    image_decoder_config,
)
from ml_collections import config_dict


def get_config(config_string: str = None):
    config = get_base_config(config_string)

    config.notes = "Offline RL"
    config.tags = ["offline_rl", "jax"]
    config.keys_to_include = {"trainer": None, "env": ["env_name"], "policy": ["name"]}

    config.data_dir = "datasets"
    config.batch_size = 64
    config.num_epochs = 1000
    config.train_frac = 1.0

    config.embedding_dim = 64
    transformer_config.embedding_dim = config.get_ref("embedding_dim")
    image_encoder_config.embedding_dim = config.get_ref("embedding_dim")

    policies = {
        "dt": config_dict.ConfigDict(
            dict(
                name="dt",
                transformer_config=transformer_config,
                image_obs=config.env.get_ref("image_obs"),
            )
        ),
        "lam": config_dict.ConfigDict(
            dict(
                name="lam",
                transformer_config=transformer_config,
                image_obs=config.env.get_ref("image_obs"),
                image_encoder_config=image_encoder_config,
                image_decoder_config=image_decoder_config,
                # vq_vae
                num_codes=8,
                code_dim=embedding_dim,
                beta=0.25,
            )
        ),
    }

    for k, v in policies.items():
        if k in config_string:
            policy_config = v

    config.policy = policy_config
    config.lr = 3e-4
    config.eps = 1e-8

    # =============================================================
    # Data collection stuff
    # =============================================================
    config.num_rollouts_collect = 1000

    config.cpu = 5
    config.gpu = 1.0  # needs more gpu here

    return config
