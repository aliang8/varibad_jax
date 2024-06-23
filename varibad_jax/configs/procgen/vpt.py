from varibad_jax.configs.procgen.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    prefix = "vpt-procgen-64x64-"
    if config_string is not None:
        config_string = prefix + config_string
    else:
        config_string = prefix
    config = get_base_config(config_string)

    config.exp_name = "vpt"

    config.data.data_type = "lapo"
    config.data.context_len = 1
    config.data.num_trajs = 100

    config.num_evals = 20
    config.num_updates = 5000
    config.save_key = "acc"
    config.best_metric = "max"

    config.model.idm.use_state_diff = False

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
        "model": {"idm": {"use_state_diff": 1}},
    }

    return config
