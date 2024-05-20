from varibad_jax.configs.xland.minigrid_gotodoor.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    config_string = "dt_lam_agent-xland-9x9"
    config = get_base_config(config_string)

    config.exp_name = "dt_lam_agent"

    config.data.data_type = "trajectories"
    config.data.context_len = 0
    config.data.num_trajs = 1000
    config.data.train_frac = 1.0
    config.data.num_trajs_per_batch = 2  # ICL hyperparameter
    config.data.resample_prompts_every_eval = True

    config.model.use_lr_scheduler = False
    config.model.latent_action_dim = 64
    config.model.lam_ckpt = "/scr/aliang80/varibad_jax/varibad_jax/results/lam/nt--1/eid-MiniGrid-GoToDoorRandomColors-R1-9x9-3/en-xland/b-0.05/code_d-64/n_codes-60"
    config.model.latent_action_decoder_ckpt = "/scr/aliang80/varibad_jax/varibad_jax/results/action_decoder/nt-1000/eid-MiniGrid-GoToDoorRandomColors-R1-9x9-3/en-xland/"

    config.embedding_dim = 128
    config.data.batch_size = 8

    config.num_evals = 40
    config.num_epochs = 2000
    config.run_eval_rollouts = True
    # config.save_key = "success"

    config.keys_to_include = {
        "seed": 1,
        "env": {"env_name": 1, "env_id": 1, "eval_env_id": 1},
        "data": {"num_trajs": 1, "num_trajs_per_batch": 1},
    }

    return config
