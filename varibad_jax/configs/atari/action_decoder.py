from varibad_jax.configs.atari.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    config_string = "latent_action_decoder-atari-64x64"
    config = get_base_config(config_string)

    config.exp_name = "action_decoder"

    # config.data.data_type = "lapo"
    config.data.data_type = "transitions"
    config.data.context_len = 1
    config.data.num_trajs = 100
    config.data.load_latent_actions = True

    config.model.use_lr_scheduler = False
    config.model.latent_action_dim = 128
    config.model.lam_ckpt = "/scr/aliang80/varibad_jax/varibad_jax/results/lam/al-False/nt-100000/eid-Pong/en-atari/b-0.05/code_d-128/n_codes-64/usd-False"

    config.num_evals = 20
    config.num_epochs = 20
    config.save_key = "acc"

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
    }

    return config
