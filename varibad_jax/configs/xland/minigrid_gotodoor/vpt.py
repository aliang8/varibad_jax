from varibad_jax.configs.xland.minigrid_gotodoor.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    config_string = "vpt-xland-9x9"
    config = get_base_config(config_string)

    config.exp_name = "vpt"

    config.data.data_type = "lapo"
    config.data.context_len = 0
    config.data.num_trajs = 1000

    config.eval_interval = 20
    config.num_epochs = 5000
    config.save_key = "acc"
    config.best_metric = "max"

    config.model.idm.use_state_diff = True

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
        "model": {"idm": {"use_state_diff": 1}},
    }

    return config
