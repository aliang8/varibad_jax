from varibad_jax.configs.xland.minigrid_gotoobject.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    config_string = "latent_action_decoder-xland-9x9"
    config = get_base_config(config_string)

    config.exp_name = "action_decoder"

    config.data.data_type = "lapo"
    config.data.context_len = 0
    #config.data.num_trajs = 20

    config.model.use_lr_scheduler = False
    config.model.latent_action_dim = 128
    config.model.lam_ckpt = "/scr/matthewh6/varibad_jax/varibad_jax/results/lam/al-False/nt-9000/eid-MiniGrid-GoToKey-R1-9x9-3/en-xland/b-0.05/code_d-128/n_codes-64/usd-True"

    config.num_evals = 20
    config.num_epochs = 2000
    config.save_key = "acc"

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
    }

    return config