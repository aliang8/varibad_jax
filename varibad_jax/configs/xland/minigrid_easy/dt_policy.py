from varibad_jax.configs.offline_config import get_config as get_offline_config


def get_config(config_string: str = None):
    config_string = "dt_agent-xland-5x5"
    config = get_offline_config(config_string)

    config.exp_name = "dt_agent"
    config.visualize_rollouts = True

    config.data.dataset_name = "minigrid_easy"
    config.data.data_type = "trajectories"
    config.data.num_trajs = -1
    config.data.train_frac = 0.9

    config.env.env_id = "XLand-MiniGridCustom-R1-7x7"
    config.env.num_episodes_per_rollout = 1
    config.env.steps_per_rollout = 20

    config.model.use_lr_scheduler = False
    config.embedding_dim = 64

    config.num_rollouts_collect = 10_000
    config.eval_interval = 50
    config.num_epochs = 500
    config.run_eval_rollouts = True

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1, "steps_per_rollout": 1},
        "data": {"num_trajs": 1},
    }

    return config
