from varibad_jax.configs.offline_config import get_config as get_offline_config


def get_config(config_string: str = None):
    config_string = "vpt_bc-xland-5x5"
    config = get_offline_config(config_string)

    config.exp_name = "vpt_bc"
    config.visualize_rollouts = True

    config.data.dataset_name = "minigrid_easy"
    config.data.data_type = "lapo"
    config.data.context_len = 0
    config.data.num_trajs = -1

    config.env.env_id = "XLand-MiniGridCustom-R1-7x7"
    config.env.num_episodes_per_rollout = 1
    config.env.steps_per_rollout = 20

    config.num_rollouts_collect = 10_000
    config.eval_interval = 10
    config.num_epochs = 100

    config.model.vpt_idm_ckpt = ""

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1, "steps_per_rollout": 1},
        "data": {"num_trajs": 1},
    }

    return config
