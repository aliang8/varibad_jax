from varibad_jax.configs.base_config import get_config as get_base_config
from varibad_jax.configs.model_configs import transformer_config, image_encoder_configs
from ml_collections import config_dict


def get_config(config_string: str = None):
    config = get_base_config(config_string)

    # =============================================================
    # VariBAD VAE configuration
    # =============================================================
    vae_config = config_dict.ConfigDict()
    vae_config.lr = 1e-3
    vae_config.buffer_size = 100_000
    vae_config.trajs_per_batch = 25
    vae_config.pretrain_len = 100_000  # number of environment steps to pretrain VAE

    # number of VAE updates per policy update
    vae_config.num_vae_updates = 3

    vae_config.kl_weight = 1e-2
    vae_config.use_kl_scheduler = False
    vae_config.max_grad_norm = 2.0
    vae_config.eps = 1e-8

    vae_config.kl_to_fixed_prior = False
    vae_config.subsample_elbos = 0
    vae_config.subsample_decode_timesteps = 0

    vae_config.latent_dim = 5

    # Reward prediction
    vae_config.decode_rewards = True
    vae_config.rew_recon_weight = 1.0
    vae_config.embedding_dim = 8

    transformer_config.encode_separate = False

    if "xland" in config_string:
        image_encoder_config = None
        for k, v in image_encoder_configs.items():
            if k in config_string:
                image_encoder_config = v
                break
        if image_encoder_config is None:
            raise ValueError("No image encoder config found for the given config string")
            
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
    policy_config = config_dict.ConfigDict()
    policy_config.image_encoder_config = image_encoder_config
    policy_config.image_obs = config.env.get_ref("image_obs")
    policy_config.pass_state_to_policy = True
    policy_config.pass_latent_to_policy = True
    policy_config.pass_belief_to_policy = False
    policy_config.pass_task_to_policy = False
    policy_config.use_hyperx_bonuses = False
    policy_config.mlp_layers = [32, 32]
    policy_config.actor_activation_function = "tanh"
    policy_config.algo = "ppo"
    policy_config.optimizer = "adam"
    policy_config.num_epochs = 2
    policy_config.num_minibatch = 4
    policy_config.clip_eps = 0.05
    policy_config.lr = 7e-4
    policy_config.eps = 1e-8
    policy_config.value_loss_coeff = 0.5
    policy_config.entropy_coeff = 0.01
    policy_config.gamma = 0.95
    policy_config.use_gae = True
    policy_config.tau = 0.95
    policy_config.max_grad_norm = 0.5
    policy_config.embedding_dim = 16
    policy_config.anneal_lr = False
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
            )
        )
    )
    return config
