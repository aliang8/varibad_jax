from varibad_jax.configs.base_config import get_config as get_base_config
from ml_collections import config_dict


def get_config(config_string: str = None):
    config = get_base_config(config_string)

    # =============================================================
    # VariBAD VAE configuration
    # =============================================================
    vae_config = config_dict.ConfigDict()
    vae_config.image_obs = config.env.get_ref("image_obs")

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

    # encoder specific configs
    encoder = {
        "lstm": config_dict.ConfigDict(
            dict(name="lstm", lstm_hidden_size=64, batch_first=False)
        ),
        "transformer": config_dict.ConfigDict(
            dict(
                name="transformer",
                hidden_dim=64,
                num_heads=8,
                num_layers=3,
                attn_size=32,
                widening_factor=4,
                dropout_rate=0.1,
                max_timesteps=1000,
                encode_separate=False,  # encode (s,a,r) as separate tokens
            )
        ),
    }

    for k, v in encoder.items():
        if k in config_string:
            encoder_config = v
            break

    encoder_config.embedding_dim = 8
    encoder_config.image_obs = config.env.get_ref("image_obs")

    # decoder specific kwargs
    decoder_config = config_dict.ConfigDict(
        dict(
            image_obs=config.env.get_ref("image_obs"),
            input_action=False,
            input_prev_state=False,
            embedding_dim=encoder_config.get_ref("embedding_dim"),
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
    policy_config.image_obs = config.env.get_ref("image_obs")
    policy_config.pass_state_to_policy = True
    policy_config.pass_latent_to_policy = True
    policy_config.pass_belief_to_policy = False
    policy_config.pass_task_to_policy = False
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
    config.policy = policy_config

    config.notes = "VariBAD JAX"
    config.tags = ["varibad", "jax"]
    config.keys_to_include = {
        "env": ["env_name"],
        "policy": ["algo", "pass_latent_to_policy"],
        "vae": ["num_vae_updates"],
    }

    config.cpu = 5
    config.gpu = 0.2

    return config
