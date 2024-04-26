from varibad_jax.configs.base_config import get_config as get_base_config
from varibad_jax.configs.model_configs import transformer_config, image_encoder_configs
from ml_collections import config_dict


def get_config(config_string: str = None):
    config = get_base_config(config_string)

    # =============================================================
    # VariBAD VAE configuration
    # =============================================================
    vae_config = config_dict.ConfigDict(
        dict(
            name="vae",
            lr=1e-3,
            buffer_size=100_000,
            trajs_per_batch=25,
            pretrain_len=100_000,  # number of environment steps to pretrain VAE
            num_vae_updates=3,  # number of VAE updates per policy update
            anneal_lr=False,
            kl_weight=1e-2,
            use_kl_scheduler=False,
            max_grad_norm=2.0,
            eps=1e-8,
            kl_to_fixed_prior=False,
            subsample_elbos=0,
            subsample_decode_timesteps=0,
            latent_dim=5,
            decode_rewards=True,
            rew_recon_weight=1.0,
            embedding_dim=8,
        )
    )
    transformer_config.encode_separate = False

    if "xland" in config_string:
        image_encoder_config = None
        for k, v in image_encoder_configs.items():
            if k in config_string:
                image_encoder_config = v
                break
        if image_encoder_config is None:
            raise ValueError(
                "No image encoder config found for the given config string"
            )

        image_encoder_config.embedding_dim = vae_config.get_ref("embedding_dim")
    else:
        image_encoder_config = None

    # encoder specific configs
    encoder = {
        "lstm": config_dict.ConfigDict(
            dict(
                name="lstm",
                lstm_hidden_size=64,
                batch_first=False,
                image_encoder_config=image_encoder_config,
            )
        ),
        "transformer": transformer_config,
    }

    encoder_config = None
    for k, v in encoder.items():
        if k in config_string:
            encoder_config = v
            break

    encoder_config.embedding_dim = vae_config.get_ref("embedding_dim")
    encoder_config.image_obs = config.env.get_ref("image_obs")

    # decoder specific kwargs
    decoder_config = config_dict.ConfigDict(
        dict(
            image_obs=config.env.get_ref("image_obs"),
            input_action=False,
            input_prev_state=False,
            embedding_dim=vae_config.get_ref("embedding_dim"),
            image_encoder_config=image_encoder_config,
            layer_sizes=[32, 32],
        )
    )

    vae_config.encoder = encoder_config
    vae_config.decoder = decoder_config

    config.vae = vae_config

    # =============================================================
    # Policy configs
    # =============================================================
    policy_config = config_dict.ConfigDict(
        dict(
            image_encoder_config=image_encoder_config,
            image_obs=config.env.get_ref("image_obs"),
            latent_dim=vae_config.get_ref("latent_dim"),
            pass_state_to_policy=True,
            pass_latent_to_policy=True,
            pass_belief_to_policy=False,
            pass_task_to_policy=False,
            use_hyperx_bonuses=False,
            mlp_layers=[32, 32],
            actor_activation_function="tanh",
            algo="ppo",
            name="ppo",
            optimizer="adam",
            num_epochs=2,
            num_minibatch=4,
            clip_eps=0.05,
            lr=7e-4,
            eps=1e-8,
            value_loss_coeff=0.5,
            entropy_coeff=0.01,
            gamma=0.95,
            use_gae=True,
            tau=0.95,
            max_grad_norm=0.5,
            embedding_dim=16,
            anneal_lr=False,
        )
    )
    config.policy = policy_config

    config.notes = "VariBAD JAX"
    config.tags = ["varibad", "jax"]
    config.keys_to_include = {
        "trainer": None,
        "env": ["env_name"],
        "policy": ["algo", "pass_latent_to_policy"],
        "vae": ["num_vae_updates", "embedding_dim"],
    }

    config.cpu = 5
    config.gpu = 0.2

    # =============================================================
    # HyperX configs
    # =============================================================
    config.hyperx = config_dict.ConfigDict(
        dict(
            lr=1e-3,
            eps=1e-8,
            rnd_weight=1.0,
            vae_recon_weight=1.0,
            latent_dim=vae_config.get_ref("latent_dim"),
            rnd=dict(
                image_obs=config.env.get_ref("image_obs"),
                embedding_dim=vae_config.get_ref("embedding_dim"),
                rnd_output_dim=32,
                layer_sizes=[32, 32],
            ),
        )
    )
    return config
