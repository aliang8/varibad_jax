from varibad_jax.configs.xland.minigrid_gotoobject.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    config_string = "lam_agent-xland-9x9"
    config = get_base_config(config_string)

    # config.exp_name = "lam_agent"
    config.exp_name = "la_bc"
    config.data.data_type = "lapo"
    config.data.context_len = 0
    config.data.num_trajs = 500

    config.model.use_lr_scheduler = False
    # config.model.latent_action_dim = 64
    config.model.latent_action_dim = 128
    config.model.lam_ckpt = "/scr/matthewh6/varibad_jax/varibad_jax/results/lam/al-False/nt-9000/eid-MiniGrid-GoToDoor-R1-9x9-3/en-xland/b-0.05/code_d-128/n_codes-64/usd-True"
    # config.model.latent_action_decoder_ckpt = "/scr/aliang80/varibad_jax/varibad_jax/results/action_decoder/nt-20/eid-MiniGrid-GoToDoorRandomColors-R1-9x9-3/en-xland/"
    config.model.latent_action_decoder_ckpt = "/scr/matthewh6/varibad_jax/varibad_jax/results/action_decoder/nt-20/eid-MiniGrid-GoToDoor-R1-9x9-3/en-xland"
    config.model.idm_nt = 20

    config.eval_interval = 40
    config.num_epochs = 2000
    config.run_eval_rollouts = True

    config.keys_to_include = {
        "seed": 1,
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
        "model": {"idm_nt": 1},
    }

    return config
