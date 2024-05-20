from varibad_jax.configs.xland.minigrid_easy.base import get_config as get_base_config


def get_config(config_string: str = None):
    config_string = "vpt_bc-xland-7x7"
    config = get_base_config(config_string)

    config.exp_name = "vpt_bc"

    config.data.data_type = "lapo"
    config.data.context_len = 0
    config.data.num_trajs = -1

    config.eval_interval = 40
    config.num_epochs = 2000

    config.model.vpt_idm_ckpt = "/scr/aliang80/varibad_jax/varibad_jax/results/vpt/nt-1000/eid-XLand-MiniGrid-TwoGoals-R1-7x7-3/en-xland/steps-30"

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1, "steps_per_rollout": 1},
        "data": {"num_trajs": 1},
    }

    return config
