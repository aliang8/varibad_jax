from varibad_jax.configs.offline_config import get_config as get_offline_rl_config


def get_config(config_string: str = None):
    config = get_offline_rl_config(config_string)
    config.exp_name = ""
    config.visualize_rollouts = True

    config.data.dataset_name = "minigrid_easy"
    config.env.env_id = "XLand-MiniGrid-TwoGoals-R1-7x7-3"
    config.env.eval_env_id = "XLand-MiniGrid-TwoGoals-R1-7x7-3"
    config.env.num_episodes_per_rollout = 1
    config.env.steps_per_rollout = 30
    config.env.full_observability = True

    config.num_rollouts_collect = 10_000

    return config
