from varibad_jax.configs.base_config import get_config as get_base_config
from ml_collections import config_dict


def get_config(config_string: str = None):
    config = get_base_config(config_string)

    config.notes = "Offline RL"
    config.tags = ["offline_rl", "jax"]
    config.keys_to_include = {"env": ["env_name"], "policy": ["name"]}

    config.data_dir = "datasets"
    config.batch_size = 64
    config.num_epochs = 1000
    config.train_frac = 1.0

    embedding_dim = 64

    transformer_config = config_dict.ConfigDict(
        dict(
            embedding_dim=embedding_dim,
            hidden_dim=64,
            num_heads=8,
            num_layers=3,
            attn_size=32,
            widening_factor=4,
            dropout_rate=0.1,
            max_timesteps=1000,
            encode_separate=True,  # encode (s,a,r) as separate tokens
        )
    )
    image_encoder_config = config_dict.ConfigDict(
        dict(
            # embedding_dim=embedding_dim,
            # output_channels=[16, 32, 64],
            # kernel_shapes=[2, 2],
            # padding="VALID",
            out_channels=embedding_dim,
            downscale_level=3,
            res_layers=2,
            kernel_size=5,
        )
    )
    image_decoder_config = config_dict.ConfigDict(
        dict(
            in_channels=embedding_dim,
            out_channels=2,  # xland specific
            upscale_level=3,
            res_layers=2,
            kernel_size=5,
        )
    )

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
