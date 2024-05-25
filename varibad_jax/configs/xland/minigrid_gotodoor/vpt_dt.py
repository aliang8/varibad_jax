from varibad_jax.configs.xland.minigrid_gotodoor.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    config_string = "vpt_icl_agent-xland-9x9"
    config = get_base_config(config_string)

    config.exp_name = "vpt_icl_agent"

    config.data.data_type = "trajectories"
    config.data.num_trajs = 5000
    config.data.num_trajs_per_batch = 1
    # config.data.resample_prompts_every_eval = True
    config.data.batch_size = 8

    config.model.use_lr_scheduler = False
    config.embedding_dim = 128

    config.eval_interval = 40
    config.num_epochs = 2000
    config.run_eval_rollouts = True
    config.gpu = 0.5

    config.keys_to_include = {
        "seed": 1,
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
    }

    return config
