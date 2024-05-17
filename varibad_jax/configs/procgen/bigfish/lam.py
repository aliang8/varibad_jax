from varibad_jax.configs.offline_config import get_config as get_offline_config


def get_config(config_string: str = None):
    config_string = "lam-procgen-64x64"
    config = get_offline_config(config_string)

    config.exp_name = "lam"
    config.visualize_rollouts = True

    config.data.dataset_name = "bigfish"
    config.data.data_type = "lapo"
    config.data.context_len = 1
    config.data.num_transitions = -1

    config.env.env_id = "bigfish"
    config.env.num_episodes_per_rollout = 1

    config.embedding_dim = 256
    config.model.idm.beta = 0.05
    config.model.idm.ema_decay = 0.999
    config.model.idm.num_codes = 60
    config.model.idm.code_dim = 64
    config.model.use_lr_scheduler = True
    config.model.warmup_steps = 50

    config.num_evals = 10
    config.num_epochs = 1000

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"context_len": 1},
        "model": {"idm": {"beta": 1, "num_codes": 1, "code_dim": 1}},
    }

    return config
