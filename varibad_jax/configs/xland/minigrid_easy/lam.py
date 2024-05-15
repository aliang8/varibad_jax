from varibad_jax.configs.offline_config import get_config as get_offline_config


def get_config(config_string: str = None):
    config_string = "lam-xland-5x5"
    config = get_offline_config(config_string)

    config.exp_name = "lam"
    config.visualize_rollouts = True

    config.data.dataset_name = "minigrid_easy"
    config.data.data_type = "lapo"
    config.data.context_len = 0
    config.data.num_trajs = 1000

    config.env.env_id = "XLand-MiniGrid-TwoGoals-R1-7x7-5"
    config.env.num_episodes_per_rollout = 1

    config.embedding_dim = 256
    config.model.idm.beta = 0.05
    config.model.idm.ema_decay = 0.999
    config.model.idm.num_codes = 60
    config.model.idm.code_dim = 128
    config.model.use_lr_scheduler = True

    config.num_rollouts_collect = 10_000
    config.num_evals = 10
    config.num_epochs = 2000

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1, "steps_per_rollout": 1},
        "data": {"num_trajs": 1},
        "model": {"idm": {"beta": 1, "num_codes": 1, "code_dim": 1}},
    }

    return config
