from varibad_jax.configs.xland.minigrid_gotodoor.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    config_string = "latent_action_decoder-xland-9x9"
    config = get_base_config(config_string)

    config.exp_name = "action_decoder"

    config.data.data_type = "lapo"
    config.data.context_len = 0
    config.data.num_trajs = 2000
    config.data.train_frac = 0.9

    config.model.use_lr_scheduler = False
    config.model.latent_action_dim = 64
    config.model.lam_ckpt = "/scr/aliang80/varibad_jax/varibad_jax/results/lam/nt--1/eid-XLand-MiniGridCustom-R1-7x7/en-xland/steps-20/b-0.05/code_d-64/n_codes-60"

    config.num_evals = 20
    config.num_epochs = 500
    config.save_key = "acc"

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
    }

    return config
