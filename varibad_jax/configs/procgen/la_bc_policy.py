from varibad_jax.configs.procgen.base import (
    get_config as get_base_config,
)


def get_config(config_string: str = None):
    config_string = "lam_agent-procgen-64x64"
    config = get_base_config(config_string)

    config.exp_name = "lam_agent"
    # config.exp_name = "la_bc_1"
    config.data.data_type = "transitions"
    config.data.context_len = 1
    config.data.num_trajs = 100
    config.data.load_latent_actions = True

    config.model.use_lr_scheduler = False
    config.model.latent_action_dim = 128

    config.eval_interval = 40
    config.num_epochs = 100
    config.run_eval_rollouts = True

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
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
