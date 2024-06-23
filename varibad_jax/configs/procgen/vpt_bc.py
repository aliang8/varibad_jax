from varibad_jax.configs.procgen.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    prefix = "vpt_bc-procgen-64x64-"
    if config_string is not None:
        config_string = prefix + config_string
    else:
        config_string = prefix
    config = get_base_config(config_string)

    config.exp_name = "vpt_bc"

    config.data.data_type = "transitions"
    config.data.num_trajs = 100
    config.data.load_latent_actions = True

    config.num_evals = 20
    config.num_updates = 50_000

    config.model.idm_nt = -1

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
        "model": {"idm_nt": 1},
    }

    return config
