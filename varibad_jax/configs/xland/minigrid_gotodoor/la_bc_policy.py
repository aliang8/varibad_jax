from varibad_jax.configs.xland.minigrid_gotodoor.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    config_string = "lam_agent-xland-9x9"
    config = get_base_config(config_string)

    config.exp_name = "lam_agent"
    config.data.data_type = "lapo"
    config.data.context_len = 0
    config.data.num_trajs = 1000

    config.model.use_lr_scheduler = False
    config.model.latent_action_dim = 64
    config.model.lam_ckpt = "/scr/aliang80/varibad_jax/varibad_jax/results/lam/nt--1/eid-MiniGrid-GoToDoorRandomColors-R1-9x9-3/en-xland/b-0.05/code_d-64/n_codes-60"
    config.model.latent_action_decoder_ckpt = "/scr/aliang80/varibad_jax/varibad_jax/results/action_decoder/nt-1000/eid-MiniGrid-GoToDoorRandomColors-R1-9x9-3/en-xland/"

    config.num_rollouts_collect = 10_000
    config.eval_interval = 40
    config.num_epochs = 2000
    config.run_eval_rollouts = True

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1, "eval_env_id": 1, "steps_per_rollout": 1},
        "data": {"num_trajs": 1},
    }

    return config
