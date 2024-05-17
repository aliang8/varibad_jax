from varibad_jax.configs.offline_config import get_config as get_offline_rl_config

# Train BC policy for simple minigrid environment


def get_config(config_string: str = None):
    config_string = "bc-xland-7x7"
    config = get_offline_rl_config(config_string)
    config.exp_name = "bc"
    config.visualize_rollouts = True

    config.data.dataset_name = "minigrid_easy"
    config.data.data_type = "transitions"
    config.data.num_trajs = 5000

    config.env.env_id = "XLand-MiniGrid-TwoGoals-R1-7x7-3"
    config.env.num_episodes_per_rollout = 1
    config.env.steps_per_rollout = 30
    config.env.full_observability = True

    config.model.policy.pass_task_to_policy = True

    config.num_rollouts_collect = 10_000
    config.num_evals = 40
    config.num_epochs = 1000

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1, "steps_per_rollout": 1},
        "data": {"num_trajs": 1},
    }

    return config
