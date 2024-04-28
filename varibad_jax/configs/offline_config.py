from pathlib import Path
from varibad_jax.configs.base_config import get_config as get_base_config
from varibad_jax.configs.model_configs import (
    transformer_config,
    image_encoder_configs,
    image_decoder_config,
)
from ml_collections.config_dict import ConfigDict


def get_config(config_string: str = None):
    config = get_base_config(config_string)

    config.trainer = "offline"
    config.notes = "Offline RL"
    config.tags = ["offline_rl", "jax"]
    config.keys_to_include = {"trainer": None, "env": ["env_name"], "policy": ["name"]}

    config.data_dir = "datasets"
    config.batch_size = 128
    config.num_epochs = 1000
    config.train_frac = 1.0
    config.eval_interval = 50
    # config.save_key = ""

    config.embedding_dim = 64
    transformer_config.embedding_dim = config.get_ref("embedding_dim")

    if "xland" in config_string:
        image_encoder_config = None
        for k, v in image_encoder_configs.items():
            if k in config_string:
                image_encoder_config = v
                break

        image_encoder_config.embedding_dim = config.get_ref("embedding_dim")
    else:
        image_encoder_config = None

    policies = {
        "dt": ConfigDict(
            dict(
                name="dt",
                transformer_config=transformer_config,
                image_encoder_config=image_encoder_config,
                batch_first=True,
            )
        ),
        "lam": ConfigDict(
            dict(
                name="lam",
                transformer_config=transformer_config,
                image_encoder_config=image_encoder_config,
                image_decoder_config=image_decoder_config,
                # vq_vae
                num_codes=8,
                code_dim=config.get_ref("embedding_dim"),
                beta=0.25,
            )
        ),
        "lapo": ConfigDict(
            dict(
                name="lapo",
                idm=dict(
                    image_obs=config.env.get_ref("image_obs"),
                    image_encoder_config=ConfigDict(
                        dict(out_channels=[16, 32, 32], out_features=256)
                    ),
                    # vq_vae
                    num_codes=4,
                    code_dim=16,
                    beta=0.05,
                    ema_decay=0.999,
                    layer_sizes=[32, 32],
                    latent_action_dim=16,
                ),
                fdm=dict(
                    image_obs=config.env.get_ref("image_obs"),
                    image_encoder_config=image_encoder_config,
                    image_decoder_config=image_decoder_config,
                ),
            )
        ),
        "lapo_agent": ConfigDict(
            dict(
                name="lapo_bc_agent",
                policy=dict(
                    lapo_model_ckpt="/home/anthony/varibad_jax/varibad_jax/results/en-xland_pn-lapo_t-offline",
                    image_obs=config.env.get_ref("image_obs"),
                    pass_latent_to_policy=False,
                    pass_task_to_policy=False,
                    image_encoder_config=image_encoder_config,
                    embedding_dim=16,
                    mlp_layers=[128, 128],
                    latent_action_dim=16,
                    gaussian_policy=False,
                ),
            )
        ),
    }

    policy_config = ConfigDict()
    for k, v in policies.items():
        if k in config_string:
            policy_config = v

    config.policy = policy_config
    config.policy.anneal_lr = False
    config.policy.lr = 3e-4
    config.policy.eps = 1e-8

    # =============================================================
    # Data collection stuff
    # =============================================================
    config.num_rollouts_collect = 10_000

    config.cpu = 5
    config.gpu = 1.0  # needs more gpu here

    return config
