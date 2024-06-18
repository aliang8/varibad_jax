from varibad_jax.configs.xland.minigrid_gotoobject.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    config_string = "dt_agent-xland-9x9"
    config = get_base_config(config_string)

    config.exp_name = "dt_agent"
    config.visualize_rollouts = True

    config.data.data_type = "trajectories"
    config.data.num_trajs = 1000
    config.data.train_frac = 1.0
    config.data.num_trajs_per_batch = 2
    config.data.resample_prompts_every_eval = True

    config.model.use_lr_scheduler = False
    config.model.policy.task_conditioning = False
    config.model.policy.demo_conditioning = True
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
