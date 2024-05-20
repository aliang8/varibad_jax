from pathlib import Path
from varibad_jax.configs.base_config import get_config as get_base_config
from varibad_jax.configs.model_configs import (
    transformer_config,
    image_encoder_configs,
    image_decoder_configs,
)
from ml_collections.config_dict import ConfigDict


def get_config(config_string: str = None):
    config = get_base_config(config_string)

    config.trainer = "offline"
    config.notes = "Offline RL"
    config.tags = ["offline_rl", "jax"]
    config.keys_to_include = {"trainer": None, "env": ["env_name"], "model": ["name"]}

    config.data = ConfigDict(
        dict(
            data_dir="datasets",
            batch_size=128,
            dataset_name="5x5",
            num_trajs=-1,
            train_frac=1.0,
            data_type="trajectories",
            # for training LAPO
            context_len=1,
            # ICL - Raparthy et. al hyperparameters
            num_trajs_per_batch=1,
            burstiness=0.5,
            holdout_tasks=False,
            resample_prompts_every_eval=False,
        )
    )

    config.num_epochs = 1000
    config.eval_interval = 50
    # config.save_key = ""

    config.embedding_dim = 64

    transformer_config.embedding_dim = config.get_ref("embedding_dim")

    if "xland" in config_string or "procgen" in config_string:
        image_encoder_config = None
        for k, v in image_encoder_configs.items():
            if k in config_string:
                image_encoder_config = v
                break

        image_decoder_config = None
        for k, v in image_decoder_configs.items():
            if k in config_string:
                image_decoder_config = v
                break

        image_encoder_config.embedding_dim = config.get_ref("embedding_dim")
    else:
        image_encoder_config = None
        image_decoder_config = None

    dt_config = ConfigDict(
        dict(
            name="dt",
            transformer_config=transformer_config,
            image_encoder_config=image_encoder_config,
            batch_first=True,
            use_rtg=False,
            image_obs=config.env.get_ref("image_obs"),
            task_conditioning=False,
            demo_conditioning=True,
        )
    )
    bc_config = ConfigDict(
        dict(
            image_obs=config.env.get_ref("image_obs"),
            image_encoder_config=image_encoder_config,
            pass_latent_to_policy=False,
            pass_task_to_policy=True,
            embedding_dim=config.get_ref("embedding_dim"),
            mlp_layer_sizes=[128, 128],
            gaussian_policy=False,
            use_rnn_policy=False,
            demo_conditioning=False,
        )
    )

    lam_ckpt = "/home/anthony/varibad_jax/varibad_jax/results/lam/en-gridworld/pn-lam/t-offline"
    latent_action_decoder_ckpt = "/home/anthony/varibad_jax/varibad_jax/results/action_decoder/en-gridworld/pn-latent_action_decoder/t-offline"

    models = {
        "bc": ConfigDict(
            dict(name="bc", image_obs=config.env.get_ref("image_obs"), policy=bc_config)
        ),
        "dt_agent": ConfigDict(
            dict(
                name="dt_agent",
                image_obs=config.env.get_ref("image_obs"),
                policy=dt_config,
            )
        ),
        "lam": ConfigDict(
            dict(
                name="lam",
                idm=dict(
                    image_obs=config.env.get_ref("image_obs"),
                    # image_encoder_config=ConfigDict(
                    #     dict(
                    #         out_channels=[16, 32, 32], out_features=256, impala_scale=4
                    #     )
                    # ),
                    image_encoder_config=image_encoder_config,
                    use_transformer=False,
                    transformer_config=transformer_config,
                    embedding_dim=config.get_ref("embedding_dim"),
                    # vq_vae
                    num_codes=6,
                    code_dim=16,
                    beta=0.25,
                    ema_decay=0.99,
                    policy_mlp_sizes=[128, 128],
                    state_embed_mlp_sizes=[128, 128],
                ),
                fdm=dict(
                    image_obs=config.env.get_ref("image_obs"),
                    image_encoder_config=image_encoder_config,
                    image_decoder_config=image_decoder_config,
                    embedding_dim=config.get_ref("embedding_dim"),
                    decoder_mlp_sizes=[128, 128],
                    use_transformer_idm=False,
                ),
                beta_loss_weight=1.0,
                context_len=config.data.get_ref("context_len"),
            )
        ),
        "latent_action_decoder": ConfigDict(
            dict(
                name="latent_action_decoder",
                image_obs=config.env.get_ref("image_obs"),
                lam_ckpt=lam_ckpt,
                latent_action_dim=16,
                context_len=config.data.get_ref("context_len"),
                mlp_layer_sizes=[128, 128, 128],
            )
        ),
        "vpt": ConfigDict(
            dict(
                name="vpt",
                idm=dict(
                    image_obs=config.env.get_ref("image_obs"),
                    # image_encoder_config=ConfigDict(
                    #     dict(
                    #         out_channels=[16, 32, 32], out_features=256, impala_scale=4
                    #     )
                    # ),
                    gaussian_policy=False,
                    image_encoder_config=image_encoder_config,
                    use_transformer=False,
                    transformer_config=transformer_config,
                    embedding_dim=config.get_ref("embedding_dim"),
                ),
                image_obs=config.env.get_ref("image_obs"),
                context_len=config.data.get_ref("context_len"),
                mlp_layer_sizes=[128, 128],
            )
        ),
        "vpt_bc": ConfigDict(
            dict(
                name="vpt_bc",
                image_obs=config.env.get_ref("image_obs"),
                policy=bc_config,
            )
        ),
        "lam_agent": ConfigDict(
            dict(
                name="lam_agent",
                image_obs=config.env.get_ref("image_obs"),
                policy=bc_config,
                image_encoder_config=image_encoder_config,
                latent_action_dim=16,
                context_len=config.data.get_ref("context_len"),
                lam_ckpt=lam_ckpt,
                latent_action_decoder_ckpt=latent_action_decoder_ckpt,
            )
        ),
        "dt_lam_agent": ConfigDict(
            dict(
                name="dt_lam_agent",
                policy=dt_config,
                image_obs=config.env.get_ref("image_obs"),
                image_encoder_config=image_encoder_config,
                latent_action_dim=16,
                lam_ckpt=lam_ckpt,
                latent_action_decoder_ckpt=latent_action_decoder_ckpt,
            )
        ),
    }

    model_config = ConfigDict()
    for k, v in models.items():
        if k in config_string:
            model_config = v

    config.model = model_config

    config.model.use_lr_scheduler = False
    config.model.warmup_steps = 10_000
    config.model.lr = 3e-4
    config.model.eps = 1e-8

    # =============================================================
    # Data collection stuff
    # =============================================================
    config.num_rollouts_collect = 50_000

    config.cpu = 5
    config.gpu = 0.2  # needs more gpu here

    return config
