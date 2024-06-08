from varibad_jax.configs.procgen.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    config_string = "lam_agent-procgen-64x64"
    config = get_base_config(config_string)

    config.exp_name = "lam_agent"
    # config.exp_name = "la_bc_1"
    config.data.data_type = "lapo"
    config.data.context_len = 1
    config.data.num_trajs = 500
    config.data.load_latent_actions = True

    config.model.use_lr_scheduler = False
    config.model.latent_action_dim = 128
    config.model.lam_ckpt = "/scr/aliang80/varibad_jax/varibad_jax/results/lam/al-False/nt-100000/eid-bigfish/en-procgen/b-0.05/code_d-128/n_codes-64/usd-False"
    config.model.latent_action_decoder_ckpt = "/scr/aliang80/varibad_jax/varibad_jax/results/action_decoder/nt-500/eid-bigfish/en-procgen/"
    # config.model.idm_nt = 20

    config.eval_interval = 40
    config.num_epochs = 100
    config.run_eval_rollouts = True

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1, "add_labelling": 1},
        # "model": {
        #     "idm": {
        #         "beta": 1,
        #         "num_codes": 1,
        #         "code_dim": 1,
        #         "use_state_diff": 1,
        #     }
        # },
    }

    return config
