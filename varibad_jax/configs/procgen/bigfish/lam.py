from varibad_jax.configs.offline_config import get_config as get_offline_config


def get_config(config_string: str = None):
    config_string = "lam-procgen-64x64"
    config = get_offline_config(config_string)

    config.exp_name = "lam"

    config.data.data_type = "lapo"
    config.data.context_len = 1
    config.data.num_trajs = 100000
    config.data.add_labelling = False
    config.data.num_labelled = 10

    config.embedding_dim = 256
    config.model.idm.beta = 0.05
    config.model.idm.ema_decay = 0.999
    # config.model.idm.beta = 0.25
    # config.model.idm.ema_decay = 0.99
    config.model.idm.num_codes = 64
    config.model.idm.code_dim = 128
    config.model.idm.num_codebooks = 2
    config.model.idm.num_discrete_latents = 4
    config.model.idm.epsilon = 1e-5
    config.model.idm.normalize_pred = True

    # need trajectory data to use transformer IDM
    # config.data.data_type = "trajectories"
    # config.model.idm.use_transformer = True
    # config.model.fdm.use_transformer_idm = True

    config.model.idm.use_state_diff = False
    config.model.use_lr_scheduler = True

    config.num_evals = 10
    config.num_epochs = 20

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1, "add_labelling": 1},
        "model": {
            "idm": {
                "beta": 1,
                "num_codes": 1,
                "code_dim": 1,
                "use_state_diff": 1,
            }
        },
    }

    return config
