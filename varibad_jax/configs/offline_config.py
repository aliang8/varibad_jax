from varibad_jax.configs.base_config import get_config as get_base_config
from ml_collections import config_dict


def get_config(config_string: str = None):
    config = get_base_config(config_string)

    config.data_dir = "datasets"
    config.batch_size = 64
    config.num_epochs = 1000
    config.train_frac = 1.0

    transformer_config = config_dict.ConfigDict(
        dict(
            embedding_dim=64,
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
    config.policy = config_dict.ConfigDict(
        dict(
            name="dt",
            transformer_config=transformer_config,
            image_obs=config.env.get_ref("image_obs"),
        )
    )
    config.lr = 3e-4
    config.eps = 1e-8
    return config
