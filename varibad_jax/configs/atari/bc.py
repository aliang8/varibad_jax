from varibad_jax.configs.atari.base import (
    get_config as get_base_config,
)

# Train BC policy for simple minigrid environment


def get_config(config_string: str = None):
    config_string = "bc-atari-84x84"
    config = get_base_config(config_string)
    config.exp_name = "bc"

    config.data.data_type = "transitions"
    config.data.num_trajs = 100

    config.model.policy.pass_task_to_policy = False

    config.num_evals = 10
    config.num_epochs = 150

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
    }

    return config
