from varibad_jax.configs.offline_config import get_config as get_offline_rl_config


def get_config(config_string: str = None):
    config = get_offline_rl_config(config_string)
    config.exp_name = ""
    config.visualize_rollouts = True

    config.env.ruleset_id = -1
    config.data.dataset_name = "rlu_atari_checkpoints_ordered"
    config.env.env_id = "Pong"
    config.env.eval_env_ids = ("Pong",)

    config.env.num_episodes_per_rollout = 1
    config.env.steps_per_rollout = 1000
    # config.env.full_observability = True
    # config.num_rollouts_collect = 10_000

    return config
