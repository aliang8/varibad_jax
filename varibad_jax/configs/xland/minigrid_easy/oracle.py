from varibad_jax.configs.rl_config import get_config as get_rl_config

# Train RL policy for simple minigrid environment


def get_config(config_string: str = None):
    config_string = "lstm-xland-7x7"
    config = get_rl_config(config_string)
    config.exp_name = "oracle"
    config.visualize_rollouts = True

    config.env.env_id = "XLand-MiniGrid-TwoGoals-R1-7x7-3"
    config.env.num_episodes_per_rollout = 1
    config.env.steps_per_rollout = 30
    config.num_frames = 20_000_000
    config.env.full_observability = True

    config.model.entropy_coeff = 0.05
    config.model.policy.pass_task_to_policy = True

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1, "steps_per_rollout": 1}
    }

    return config
