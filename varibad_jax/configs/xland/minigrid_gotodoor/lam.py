from varibad_jax.configs.xland.minigrid_gotodoor.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    config_string = "lam-xland-9x9"
    config = get_base_config(config_string)

    config.exp_name = "lam"

    config.data.data_type = "lapo"
    config.data.context_len = 0
    config.data.num_trajs = -1

    config.embedding_dim = 256
    # config.model.idm.beta = 0.05
    # config.model.idm.ema_decay = 0.999
    config.model.idm.beta = 0.05
    config.model.idm.ema_decay = 0.999
    config.model.idm.num_codes = 60
    config.model.idm.code_dim = 64
    config.model.use_lr_scheduler = True

    config.num_evals = 20
    config.num_epochs = 500

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
        "model": {"idm": {"beta": 1, "num_codes": 1, "code_dim": 1}},
    }

    return config
