from varibad_jax.configs.procgen.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    config_string = "dt_agent-procgen-64x64"
    config = get_base_config(config_string)

    config.exp_name = "dt_agent"
    config.visualize_rollouts = True

    config.data.data_type = "trajectories"
    config.data.num_trajs = 100
    config.data.train_frac = 1.0
    config.data.num_trajs_per_batch = 1
    config.data.context_window = 10

    config.model.use_lr_scheduler = False
    config.embedding_dim = 128

    config.eval_interval = 40
    config.num_epochs = 2000
    config.run_eval_rollouts = True

    config.keys_to_include = {
        "seed": 1,
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
    }

    return config
