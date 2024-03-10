"""
blaze run -c opt experimental/posterior_transformer:evaluate -- \
    --config=configs/varibad_base.py \
    --config.exp.results_log_dir= \
"""

from collections.abc import Sequence
import functools
from functools import partial
import os
import time
from typing import Optional

from absl import app
from absl import flags
from absl import logging
import chex
import flax.linen as nn
from flax.training.train_state import TrainState
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
from ml_collections.config_flags import config_flags
import numpy as np
import optax
from sklearn.metrics import precision_recall_fscore_support
import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tensorflow_probability.substrates.jax.distributions

from google3.experimental.posterior_transformer.models.helpers import decode
from google3.experimental.posterior_transformer.models.update import compute_reward_pred_loss
from google3.experimental.posterior_transformer.models.utils import compute_kl_prev_posterior, reparameterize
from google3.experimental.posterior_transformer.rollout import eval_rollout, eval_rollout_transformer
from google3.experimental.posterior_transformer.utils.general_utils import CheckpointSaver, convert_to_tfp, get_prior, gumbel_softmax
from google3.experimental.posterior_transformer.utils.viz_utils import make_trajectory_video_subplots, plot_kl_term, plot_latents, plot_reward_reconstruction, plot_to_image
from google3.learning.brain.research.jax.jaxboard import jaxboard
from google3.pyglib import gfile
from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing
from google3.experimental.posterior_transformer.agents.rule_based import RuleBasedPolicy


FLAGS = flags.FLAGS
if not hasattr(FLAGS, 'config'):
  config_flags.DEFINE_config_file(
      'config',
      'experimental/posterior_transformer/configs/varibad_base.py',
      'Training configuration.',
  )


def evaluate(
    config: ConfigDict,
    ts_policy: TrainState,
    ts_vae: TrainState,
    rng_seq: hk.PRNGSequence,
    video_dir: str,
    iter_idx: int = 0,
    goal_transition_matrix: Optional[np.ndarray] = None,
):
  rollout_fn = (
      eval_rollout_transformer
      if 'transformer' in config.vae.model_cls
      else eval_rollout
  )

  decode_fn = partial(jax.jit, static_argnames='config')(decode.apply)

  if config.policy.rule_based:
    policy = RuleBasedPolicy(config)  # for rule based agent
  else:
    policy = None

  # rollout is a dictionary
  returns_per_episode, rollouts = rollout_fn(
      config=config,
      ts_policy=ts_policy,
      ts_vae=ts_vae,
      rng_seq=rng_seq,
      goal_transition_matrix=goal_transition_matrix,
      num_rollouts=config.exp.num_eval_rollouts,
      policy=policy,
  )

  extras = {}

  if config.vae.prior_type == 'gaussian':
    latent_mean, latent_logvar = jnp.split(rollouts['latent'], 2, axis=-1)
  else:
    latent_mean = rollouts['latent']
    latent_logvar = np.zeros_like(latent_mean)

  posterior = convert_to_tfp(config.vae.prior_type, rollouts['latent'])
  if config.vae.prior_type == 'categorical':
    latent_samples = gumbel_softmax(
        rng_key=next(rng_seq), logits=rollouts['latent'], temperature=1.0
    )
  else:
    latent_samples = posterior.sample(seed=next(rng_seq))

  if len(latent_samples.shape) == 2:
    latent_samples = latent_samples[..., jnp.newaxis]

  print(latent_samples.shape)

  T, B = latent_mean.shape[:2]
  extras['latent_mean'] = latent_mean
  extras['latent_logvar'] = latent_logvar
  extras['latent_samples'] = latent_samples

  # [T, num_rollouts, num_states]
  decode_outputs = decode_fn(
      ts_vae.params,
      next(rng_seq),
      config=FrozenConfigDict(config.vae),
      latent_samples=latent_samples,
      prev_states=rollouts['states'],
      next_states=rollouts['next_states'],
      actions=rollouts['actions'],
  )

  extras['decode_outputs'] = decode_outputs

  # [T, B]
  prior = get_prior(
      config.vae.prior_type, prior_shape=(T, B, config.vae.latent_dim)
  )
  kl_term = posterior.kl_divergence(prior)

  if len(kl_term.shape) == 3:
    kl_term = kl_term.mean(-1)

  print('kl term shape: ', kl_term.shape)
  extras['kl_term'] = kl_term

  possible_goals = rollouts['possible_goals'][:, jnp.newaxis]
  gt_goals = rollouts['goals']
  rew_target = (possible_goals[..., -1] == gt_goals[..., -1]).astype(jnp.int32)
  goal_indices = rew_target[..., jnp.newaxis]

  state_indices = None

  if 'grid' in config.env.env_name.lower():
    if config.vae.pred_type == 'next_state_reward':
      env = gym.make('GridNavi-v0', env_config=config.env)
      state_indices = rollouts['next_states'][..., 0:1].astype(jnp.int32)

  # Decode reward and state. The reward here should be raw logits.
  # [T, B]
  rew_recon_loss = compute_reward_pred_loss(
      rew_preds=decode_outputs.rew_pred,
      rewards=rollouts['rewards'],
      pred_type=config.vae.pred_type,
      goal_indices=goal_indices,
      state_indices=state_indices,
  )
  print('rew recon loss shape: ', rew_recon_loss.shape)
  extras['rew_recon_loss'] = rew_recon_loss

  posterior = convert_to_tfp(config.vae.prior_type, rollouts['latent'])
  final_latent_belief = rollouts['latent'][-1][np.newaxis].repeat(T, 0)
  final_posterior = convert_to_tfp(config.vae.prior_type, final_latent_belief)
  kl_to_final_belief = posterior.kl_divergence(final_posterior)

  if len(kl_to_final_belief.shape) == 3:
    kl_to_final_belief = kl_to_final_belief.mean(-1)

  if config.exp.save_video:
    # Visualize agent's behavior
    logging.info(f'Saving rollout videos to: {video_dir}')
    iterations = min(
        config.exp.num_eval_rollouts, config.exp.num_eval_rollouts_save
    )

    for k, v in rollouts.items():
      print(k)
      if k not in ['infos', 'all_values']:
        print(v.shape)

    for traj_idx in range(iterations):
      filename = os.path.join(
          video_dir,
          f'rollout_env_{traj_idx:03d}_step_{iter_idx:05d}.html',
      )

      rollout = {
          k: v[:, traj_idx]
          for k, v in rollouts.items()
          if k not in ['infos', 'all_values']
      }

      # for ant environment, let's plot the trajectory in 2D to visualize
      if 'ant' in config.env.env_name.lower():
        assert 'ant_pos' in list(rollout.keys())
        ant_pos = np.array(rollout['ant_pos'])

        fig, ax = plt.subplots(figsize=(8, 8))
        (ax_plot,) = ax.plot(
            ant_pos[0, 0], ant_pos[0, 1], 'go', markersize=15, markeredgewidth=5
        )
        ax.plot(
            ant_pos[-1, 0],
            ant_pos[-1, 1],
            'rx',
            markersize=15,
            markeredgewidth=5,
        )

        def animate(idx):
          ax_plot.set_data(ant_pos[:idx, 0], ant_pos[:idx, 1])

        # add markers for the goals
        if 'square' in config.env.env_name.lower():
          goals = rollout['goals'][rollout['done_mdp']]

          for idx, goal in enumerate(goals):
            ax.plot(goal[0], goal[1], 'yx', markersize=15, markeredgewidth=5)
            ax.text(goal[0], goal[1], str(idx))

        anim = FuncAnimation(fig, animate, frames=ant_pos.shape[0], repeat=True)

        traj_viz_file = os.path.join(
            video_dir, f'ant_traj_viz_{traj_idx:03d}_step_{iter_idx:05d}.html'
        )
        with gfile.Open(traj_viz_file, 'wb') as f:
          f.write(anim.to_jshtml())

      if (
          config.vae.pred_type == 'next_state_reward'
          and 'grid' in config.env.env_name.lower()
      ):
        beliefs = nn.sigmoid(decode_outputs.rew_pred)
        beliefs = beliefs[:, traj_idx]
      else:
        beliefs = None

      make_trajectory_video_subplots(
          env_config=config.env,
          kl_term=kl_term[:-2, traj_idx],
          rew_recon_loss=rew_recon_loss[:-2, traj_idx],
          kl_to_final_belief=kl_to_final_belief[:-2, traj_idx],
          beliefs=beliefs,
          rollout=rollout,
          save=True,
          filename=filename,
          fps=config.exp.fps,
      )

      # Make plots of the latent
      fig = plt.figure(figsize=(16, 8))
      gs = fig.add_gridspec(2, 3)
      ax1 = fig.add_subplot(gs[0, 0])
      ax2 = fig.add_subplot(gs[0, 1])
      ax3 = fig.add_subplot(gs[0, 2])
      ax4 = fig.add_subplot(gs[1, 0])
      ax5 = fig.add_subplot(gs[1, 1])
      ax6 = fig.add_subplot(gs[1, 2])
      axes = [ax1, ax2, ax3, ax4, ax5, ax6]

      goals = rollout['goals']
      T = len(goals)
      x = np.arange(T)
      latent_means = extras['latent_mean'][:, traj_idx]
      latent_logvars = extras['latent_logvar'][:, traj_idx]
      latent_samples = extras['latent_samples'][:, traj_idx]
      reward_preds = extras['decode_outputs'].rew_pred[:, traj_idx]
      gt_rewards = rollout['rewards']
      std_avg = np.exp(0.5 * np.mean(latent_logvars, axis=-1))
      std = np.exp(0.5 * latent_logvars)

      all_plots = [
          [goals],
          [goals, np.mean(latent_samples, axis=-1)],
          [latent_samples[..., i] for i in range(config.vae.latent_dim)]
          + [np.mean(latent_samples, axis=-1)],
          [std[..., i] for i in range(config.vae.latent_dim)] + [std_avg],
          [latent_means[..., i] for i in range(config.vae.latent_dim)]
          + [np.mean(latent_means, axis=-1)],
          [reward_preds, gt_rewards],
      ]

      all_labels = [
          ['goal_idx'],
          ['goal_idx'],
          [f'latent_{i}' for i in range(config.vae.latent_dim)]
          + ['latent_avg'],
          [f'std_{i}' for i in range(config.vae.latent_dim)] + ['std_avg'],
          [f'mean_{i}' for i in range(config.vae.latent_dim)] + ['mean_avg'],
          ['reward', 'gt_reward'],
      ]

      all_titles = [
          'Goal Index',
          '-',
          'Latent Samples',
          'Latent Std',
          'Latent Mean',
          'Reward vs Pred',
      ]

      for ax, plots, labels, title in zip(
          axes, all_plots, all_labels, all_titles
      ):
        for plot, label in zip(plots, labels):
          ax.plot(plot, label=label)
          ax.set_title(title)

      # put some vertical lines where the goal changes
      index = np.where(rollout['done_mdp'] == 1)[0]
      goal_index_change = np.where(np.array(goals[:-1]) != np.array(goals[1:]))[
          0
      ]

      for ax in axes:
        for ind in index:
          ax.axvline(x=ind, ls='--', color='gray')
        for ind in goal_index_change:
          ax.axvline(x=ind, ls=':', color='yellow', lw=2)

        ax.legend()

      viz_file = os.path.join(
          video_dir, f'latent_viz_{traj_idx:03d}_step_{iter_idx:05d}.png'
      )
      with gfile.Open(viz_file, 'wb') as f:
        plt.savefig(f)

  return returns_per_episode, rollouts, extras


def main(argv: Sequence[str]) -> None:
  del argv

  config = FLAGS.config
  logging.info('Config: %s', config)
  logging.info('Devices: %s', jax.devices())
  num_devices = jax.device_count()
  num_local_devices = jax.local_device_count()
  logging.info(
      'Num local devices: %d, Num devices: %d', num_local_devices, num_devices
  )
  num_hosts = num_devices // num_local_devices
  assert num_hosts == 1, 'Use only 1 host for sample eval.'

  while not gfile.Exists(config.exp.results_log_dir):
    logging.info('waiting for training to start...')
    time.sleep(60)

  # Task-specific directories
  eval_dir = os.path.join(config.exp.results_log_dir, 'eval')

  # if gfile.Exists(eval_dir):
  #   logging.info(f'Path: {eval_dir} already exists, overwriting...')
  #   gfile.DeleteRecursively(eval_dir)

  eval_video_dir = os.path.join(eval_dir, 'videos')

  if not gfile.Exists(eval_dir):
    gfile.MakeDirs(eval_dir)

  if not gfile.Exists(eval_video_dir):
    gfile.MakeDirs(eval_video_dir)

  # Initialize Tensorboard.
  logging.info('Creating eval summary writer')
  sw = jaxboard.SummaryWriter(eval_dir)

  # initialize models
  rng_seq = hk.PRNGSequence(jax.random.PRNGKey(config.exp.seed))

  ckpt_dir = os.path.join(config.exp.results_log_dir, 'models')

  while not gfile.Exists(ckpt_dir) or len(gfile.ListDirectory(ckpt_dir)) == 0:
    logging.info('waiting for model ckpts')
    time.sleep(60)

  logging.info(f'loading checkpoints from: {ckpt_dir}')

  # Begin monitoring, wait for a new checkpoint to show up
  # Load the weights and evaluate the model
  previous_checkpoint = None

  while True:
    ckpter = CheckpointSaver(ckpt_dir)
    latest_checkpoint = ckpter.get_latest_ckpt()
    logging.info(f'latest checkpoint: {latest_checkpoint}')

    # Ensure we only evaluate a checkpoint 1x.
    while latest_checkpoint == previous_checkpoint:
      logging.info(
          'We already evaluated latest checkpoint: %s', latest_checkpoint
      )
      time.sleep(60)
      # need to reload this
      try:
        ckpter = CheckpointSaver(ckpt_dir)
        latest_checkpoint = ckpter.get_latest_ckpt()
      except:
        pass

    # Found new checkpoint
    assert latest_checkpoint is not None
    previous_checkpoint = latest_checkpoint

    # if not gfile.Exists(latest_checkpoint):
    #   logging.info(
    #       "skipping this checkpoint, because for some reason it doesn't exist"
    #       ' anymore'
    #   )
    #   continue

    ckpt = ckpter.load_from_ckpt(latest_checkpoint)
    if 'config' in ckpt and ckpt['config'] is not None:
      # overwrite config with the one from the experiment
      config = ConfigDict(ckpt['config'])
      logging.info('update config: ')
      logging.info(config)

    vae_ckpt, policy_ckpt = ckpt['ts_vae'], ckpt['ts_policy']
    policy_params = policy_ckpt['params']
    vae_params = vae_ckpt['params']

    if config.env.env_name == 'GridNaviAlternatingGoals-v0':
      if 'goal_transition_matrix' not in ckpt:
        raise Exception
      else:
        goal_transition_matrix = ckpt['goal_transition_matrix']
    else:
      goal_transition_matrix = None

    # this doesn't really matter
    tx_vae = optax.chain(
        # optax.clip(config.policy.max_grad_norm),
        optax.adam(config.vae.lr, eps=config.policy.eps),
    )
    ts_vae = TrainState.create(apply_fn=None, params=vae_params, tx=tx_vae)
    ts_policy = TrainState.create(
        apply_fn=None,
        params=policy_params,
        tx=tx_vae,
    )

    returns_per_episode, rollouts, _ = evaluate(
        config=config,
        ts_policy=ts_policy,
        ts_vae=ts_vae,
        rng_seq=rng_seq,
        goal_transition_matrix=goal_transition_matrix,
        video_dir=eval_video_dir,
        iter_idx=latest_checkpoint,
    )

    avg_returns_per_episode = returns_per_episode.mean(axis=0)

    eval_metrics = {
        f'returns_per_episode/avg/episode_{j}': avg_returns_per_episode[j]
        for j in range(len(avg_returns_per_episode))
    }
    eval_metrics['returns_per_episode/avg'] = returns_per_episode.mean()
    total_return = returns_per_episode.sum(axis=-1).mean()
    eval_metrics['returns_per_episode/total'] = total_return

    if 'opt_acts' in rollouts:
      pr = precision_recall_fscore_support(
          y_true=rollouts['opt_acts'].reshape(-1),
          y_pred=rollouts['actions'].reshape(-1),
          average='weighted',
          zero_division=1,
      )
      precision, recall, fb, support = pr
      eval_metrics['precision'] = precision
      eval_metrics['recall'] = recall
      eval_metrics['fbeta'] = fb

    if 'opt_rews' in rollouts:
      optimal_rets = rollouts['opt_rews'].sum(axis=0).mean()
      eval_metrics['opt_ret_total'] = optimal_rets
      eval_metrics['delta_from_opt'] = optimal_rets - total_return

    for k, v in eval_metrics.items():
      sw.scalar(f'eval/{k}', v.tolist(), step=latest_checkpoint)

    sw.flush()

    logging.info(f'Finished evaluating checkpoint: {latest_checkpoint}')


if __name__ == '__main__':
  g3_multiprocessing.handle_main(main)
