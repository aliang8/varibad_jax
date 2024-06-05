from varibad_jax.configs.rl_config import get_config as get_rl_config

# Train RL policy for simple minigrid environment


def get_config(config_string: str = None):
    config_string = "lstm-minatar-10x10"
    config = get_rl_config(config_string)
    config.exp_name = "oracle"
    config.visualize_rollouts = True

    # config.env.env_id = "Breakout-MinAtar"
    # config.env.env_id = "Asterix-MinAtar"
    # config.env.env_id = "SpaceInvaders-MinAtar"
    # config.env.env_id = "Freeway-MinAtar"
    # config.env.env_id = "Seaquest-MinAtar"

    config.env.num_episodes_per_rollout = 1
    config.env.steps_per_rollout = 1000
    config.num_frames = 20_000_000
    config.num_evals = 20
    config.env.eval_env_ids = [config.env.env_id]

    config.model.entropy_coeff = 0.05

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1, "steps_per_rollout": 1}
    }

    return config
