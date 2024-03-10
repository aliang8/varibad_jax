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
    env_config.env_id = "GridNaviJAX-v0"
    env_config.num_frames = 20_000_000  # 20M

    env_config.num_episodes_per_rollout = 4
    env_config.steps_per_rollout = 15

    # number of parallel training environments
    env_config.num_processes = 64
    env_config.normalize_rews = False
    config.env = env_config

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

    vae_config.kl_to_fixed_prior = True
    vae_config.embedding_dim = 8
    vae_config.subsample_elbos = 0
    vae_config.subsample_decode_timesteps = 0

    vae_config.latent_dim = 5
    vae_config.lstm_hidden_size = 64

    # Reward prediction
    vae_config.decode_rewards = True
    vae_config.rew_recon_weight = 1.0
    vae_config.input_action = False
    vae_config.input_prev_state = False
    vae_config.rew_decoder_layers = [32, 32]
    config.vae = vae_config

    # =============================================================
    # Policy configs
    # =============================================================
    policy_config = config_dict.ConfigDict()
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
    policy_config.entropy_coeff = 0.1
    policy_config.gamma = 0.95
    policy_config.use_gae = True
    policy_config.tau = 0.95
    policy_config.max_grad_norm = 2.0
    policy_config.embedding_dim = 32
    config.policy = policy_config

    # =============================================================
    # Experiment stuff
    # =============================================================

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
    config.tags = ["varibad", "jax", "gridworld"]

    # number of warmup steps before training
    config.warmup_steps = 10000
    # where to save experiment artifacts (videos, checkpoints, etc)
    config.root_dir = "/data/anthony/varibad_jax/varibad_jax/"
    config.exp_name = ""

    # for rollout visualization
    config.fps = 5
    config.num_eval_rollouts = 10

    # number of eval rollout videos to save
    config.save_video = True
    config.num_eval_rollouts_save = 3
    config.skip_first_eval = True

    # for saving checkpoints
    config.max_ckpt_to_keep = None

    # resume training / evaluation
    config.restore_from_ckpt = False
    config.checkpoint_step = config_dict.placeholder(int)
    return config
