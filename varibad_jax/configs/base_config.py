"""Config for posterior transformer training."""

from flax import struct
from ml_collections import config_dict
import numpy as np


def get_config(config_string: str = None):
    """Defines base config for training."""

    # =============================================================
    # Environment config
    # =============================================================
    config = config_dict.ConfigDict()

    env_config = config_dict.ConfigDict()
    env_config.num_frames = 20_000_000

    # define env specific configs
    envs = {
        "gridworld": config_dict.ConfigDict(
            dict(
                env_name="gridworld",
                env_id="GridNaviJAX-v0",
                num_episodes_per_rollout=4,
                steps_per_rollout=15,
                num_processes=16,
                normalize_rews=False,
                image_obs=False,
            )
        ),
        "xland": config_dict.ConfigDict(
            dict(
                env_name="xland",
                env_kwargs=dict(
                    view_size=5,
                    height=5,
                    width=5,
                ),
                benchmark_path="/scr/aliang80/varibad_jax/varibad_jax/envs/xland_benchmarks/test_ruleset.pkl",
                ruleset_id=0,
                env_id="XLand-MiniGrid-R1-9x9",
                num_episodes_per_rollout=4,
                steps_per_rollout=15,
                num_processes=16,
                normalize_rews=False,
                image_obs=True,
            )
        ),
    }

    for k, v in envs.items():
        if k in config_string:
            env_config.update(v)

    config.env = env_config

    # =============================================================
    # VariBAD VAE configuration
    # =============================================================
    vae_config = config_dict.ConfigDict()
    vae_config.image_obs = env_config.get_ref("image_obs")

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
            )
        ),
    }

    for k, v in encoder.items():
        if k in config_string:
            encoder_config = v
            break

    encoder_config.embedding_dim = 8
    encoder_config.image_obs = env_config.get_ref("image_obs")

    # decoder specific kwargs
    decoder_config = config_dict.ConfigDict(
        dict(
            image_obs=env_config.get_ref("image_obs"),
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
    policy_config.image_obs = env_config.get_ref("image_obs")
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

    # =============================================================
    # Experiment stuff
    # =============================================================

    config.trainer = "vae_trainer"
    config.seed = 521
    config.mode = "train"
    config.use_wb = False

    # number of updates
    config.log_level = "info"
    config.enable_jit = True
    config.log_interval = 10
    config.save_interval = 100
    config.eval_interval = 100
    config.disable_tqdm = True
    config.smoke_test = True
    config.notes = "VariBAD JAX"
    config.tags = ["varibad", "jax"]

    # number of warmup steps before training
    config.warmup_steps = 10000
    # where to save experiment artifacts (videos, checkpoints, etc)
    config.root_dir = "/scr/aliang80/varibad_jax/varibad_jax/"
    config.exp_name = "vb_jax"
    config.ray_logdir = "ray_results"

    # for rollout visualization
    config.fps = 5
    config.num_eval_rollouts = 10

    # number of eval rollout videos to save
    config.visualize_rollouts = False
    config.num_eval_rollouts_save = 3
    config.skip_first_eval = False

    # for saving checkpoints
    config.max_ckpt_to_keep = None

    # resume training / evaluation
    config.load_from_ckpt = False
    config.checkpoint_step = config_dict.placeholder(int)
    return config
