from varibad_jax.configs.xland.minigrid_gotodoor.base import (
    get_config as get_base_config,
)

# Train BC policy for simple minigrid environment


def get_config(config_string: str = None):
    config_string = "bc-xland-9x9"
    config = get_base_config(config_string)
    config.exp_name = "bc"

    config.data.data_type = "transitions"
    config.data.num_trajs = 1000

    config.model.policy.pass_task_to_policy = False

    config.num_evals = 40
    config.num_epochs = 2000
    # config.save_key = "success"

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
    }

    return config
