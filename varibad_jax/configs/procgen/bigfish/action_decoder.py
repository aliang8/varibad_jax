from varibad_jax.configs.offline_config import get_config as get_offline_config


def get_config(config_string: str = None):
    config_string = "latent_action_decoder-procgen-64x64"
    config = get_offline_config(config_string)

    config.exp_name = "action_decoder"
    config.visualize_rollouts = True

    config.data.dataset_name = "bigfish"
    config.data.data_type = "lapo"
    config.data.context_len = 1
    config.data.num_transitions = 5000

    config.env.env_id = "bigfish"
    config.env.num_episodes_per_rollout = 1

    config.model.use_lr_scheduler = False
    config.model.latent_action_dim = 64
    config.model.lam_ckpt = "/scr/aliang80/varibad_jax/varibad_jax/results/lam/nt--1/eid-XLand-MiniGridCustom-R1-7x7/en-xland/steps-20/b-0.05/code_d-64/n_codes-60"

    config.num_evals = 10
    config.num_epochs = 100
    config.save_key = "acc"

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
    }

    return config
