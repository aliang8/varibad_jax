from varibad_jax.configs.offline_config import get_config as get_offline_config


def get_config(config_string: str = None):
    config_string = "lam_agent-xland-7x7"
    config = get_offline_config(config_string)

    config.exp_name = "lam_agent"
    config.visualize_rollouts = True

    config.data.dataset_name = "minigrid_easy"
    config.data.data_type = "lapo"
    config.data.context_len = 0
    config.data.num_trajs = 1000
    config.data.train_frac = 1.0

    config.env.env_id = "XLand-MiniGrid-TwoGoals-R1-7x7-3"
    config.env.num_episodes_per_rollout = 1
    config.env.full_observability = True

    config.model.use_lr_scheduler = False
    config.model.latent_action_dim = 64
    config.model.lam_ckpt = "/scr/aliang80/varibad_jax/varibad_jax/results/lam/nt--1/eid-XLand-MiniGridCustom-R1-7x7/en-xland/steps-20/b-0.05/code_d-64/n_codes-60"
    config.model.latent_action_decoder_ckpt = "/scr/aliang80/varibad_jax/varibad_jax/results/action_decoder/nt-1000/eid-XLand-MiniGridCustom-R1-7x7/en-xland/steps-20/"

    config.num_rollouts_collect = 10_000
    config.num_evals = 20
    config.num_epochs = 5000
    config.run_eval_rollouts = True

    config.model.policy.pass_task_to_policy = True

    config.keys_to_include = {
        "env": {"env_name": 1, "env_id": 1},
        "data": {"num_trajs": 1},
    }

    return config
