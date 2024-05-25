from varibad_jax.configs.xland.minigrid_gotodoor.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    config_string = "vpt_bc-xland-9x9"
    config = get_base_config(config_string)

    config.exp_name = "vpt_bc"

    config.data.data_type = "lapo"
    config.data.context_len = 0
    config.data.num_trajs = 100

    config.eval_interval = 40
    config.num_epochs = 2000

    config.model.vpt_idm_ckpt = "/scr/aliang80/varibad_jax/varibad_jax/ray_results/vpt/vpt/nt-10/s-0/eid-MiniGrid-GoToDoorRandomColors-R1-9x9-3/en-xland/usd-True"
    config.model.idm_nt = -1

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
        "model": {"idm_nt": 1},
    }

    return config
