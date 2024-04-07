import time
from absl import logging
import jax
import chex
import haiku as hk
import jax.numpy as jnp
from xminigrid.environment import Environment, EnvParamsT
from flax.training.train_state import TrainState
from typing import Callable, Union, Tuple
import numpy as np
import wandb
import einops
import functools
from ml_collections import ConfigDict
import jax.tree_util as jtu
from xminigrid.experimental.img_obs import TILE_W_AGENT_CACHE, TILE_CACHE, TILE_SIZE
from xminigrid.core.constants import NUM_COLORS, TILES_REGISTRY, Tiles


@chex.dataclass
class RolloutStats:
    episode_return: jnp.ndarray
    length: jnp.ndarray


def run_rollouts(
    rng: jax.random.PRNGKey,
    state: Union[hk.State, Tuple[hk.State, hk.State]],
    config: ConfigDict,
    env: Environment,
    ts_policy: TrainState,
    steps_per_rollout: int,
    action_dim: int,
    wandb_run=None,
    **kwargs
) -> RolloutStats:
    logging.info("Evaluating policy rollout...")

    rng_keys = jax.random.split(rng, config.num_eval_rollouts)

    start = time.time()

    if config.trainer == "vae":
        rollout_fn = eval_rollout_with_belief_model
        rollout_kwargs = dict(
            ts_policy=ts_policy,
            action_dim=action_dim,
            steps_per_rollout=steps_per_rollout,
            **kwargs
        )
    elif config.trainer == "offline" and config.policy.name == "dt":
        rollout_fn = eval_rollout_dt
        rollout_kwargs = dict(
            ts_policy=ts_policy,
            action_dim=action_dim,
            steps_per_rollout=steps_per_rollout,
        )
    elif config.trainer == "rl":
        rollout_fn = eval_rollout
        rollout_kwargs = dict(
            ts_policy=ts_policy,
            action_dim=action_dim,
            steps_per_rollout=steps_per_rollout,
        )
    else:
        rollout_kwargs = dict()
        return {}

    rollout_kwargs.update(dict(env=env, config=config, state=state))

    # render function doesn't work with vmap
    eval_metrics, (transitions, actions) = jax.vmap(
        functools.partial(rollout_fn, **rollout_kwargs)
    )(rng_keys)

    rollout_time = time.time() - start
    fps = (config.num_eval_rollouts * steps_per_rollout) / rollout_time

    eval_metrics = {
        "episode_return": jnp.mean(eval_metrics["episode_return"]),
        "avg_length": jnp.mean(eval_metrics["length"]),
        "fps": fps,
    }
    print(eval_metrics)

    # visualize the rollouts
    if wandb_run is not None and config.visualize_rollouts:
        # imgs is N x T x H x W x C
        # we want N x T x C x H x W

        # generate the images
        videos = []
        for rollout_indx in range(config.num_eval_rollouts):
            images = []
            for step in range(steps_per_rollout):
                timestep = jtu.tree_map(lambda x: x[rollout_indx][step], transitions)
                images.append(env.render(env.env_params, timestep))
            videos.append(images)

        videos = np.array(videos)

        videos = einops.rearrange(videos, "n t h w c -> n t c h w")
        wandb_run.log({"eval_rollouts": wandb.Video(np.array(videos), fps=5)})

    return eval_metrics, transitions, actions


def eval_rollout_dt(
    rng: jax.Array,
    state: hk.State,
    env: Environment,
    config: dict,
    ts_policy: TrainState,
    action_dim: int,
    steps_per_rollout: int,
) -> RolloutStats:

    stats = RolloutStats(episode_return=0, length=0)

    rng, reset_rng = jax.random.split(rng, 2)
    xtimestep = env.reset(env.env_params, reset_rng)

    observation_shape = xtimestep.timestep.observation.shape

    # first dimension is batch
    observations = jnp.zeros((1, steps_per_rollout, *observation_shape))
    actions = jnp.zeros((1, steps_per_rollout, action_dim))
    rewards = jnp.zeros((1, steps_per_rollout, 1))
    mask = jnp.ones((1, steps_per_rollout))
    done = False

    # first return-to-go is the desired return value
    # this is probably environment / task specific
    rtg = 60.0
    rewards = rewards.at[0, 0].set(rtg)

    def _step_fn(carry, _):
        (
            rng,
            stats,
            state,
            xtimestep,
            observations,
            actions,
            rewards,
            rtg,
            mask,
            done,
        ) = carry

        rng, policy_rng = jax.random.split(rng)
        observation = xtimestep.timestep.observation
        observation = observation.astype(jnp.float32)
        observations = observations.at[0, stats.length].set(observation)
        mask = mask.at[0, stats.length].set(1.0)

        policy_output, state = ts_policy.apply_fn(
            ts_policy.params,
            state,
            policy_rng,
            states=observations,
            actions=actions,
            rewards=rewards,
            mask=mask,
            is_training=False,
        )

        # [B, T]
        action = policy_output.action
        action = action[0, stats.length]
        xtimestep = env.step(env.env_params, xtimestep, action)
        timestep = xtimestep.timestep
        reward = timestep.reward
        done = timestep.last()

        action = action.reshape(1).astype(jnp.float32)
        reward = reward.reshape(1)

        # import ipdb

        # ipdb.set_trace()
        actions = actions.at[0, stats.length].set(action)

        # return to go is the previous return minus rewad
        rtg = rtg - timestep.reward
        rewards = rewards.at[0, stats.length].set(rtg)

        stats = stats.replace(
            episode_return=stats.episode_return + timestep.reward,
            length=stats.length + 1,
        )

        return (
            rng,
            stats,
            state,
            xtimestep,
            observations,
            actions,
            rewards,
            rtg,
            mask,
            done,
        ), (timestep, action.squeeze(axis=-1))

    init_carry = (
        rng,
        stats,
        state,
        xtimestep,
        observations,
        actions,
        rewards,
        rtg,
        mask,
        done,
    )

    # transitions contains (timestep, action)
    carry, transitions = jax.lax.scan(
        _step_fn, init_carry, None, length=steps_per_rollout
    )
    stats = carry[1]
    return stats, transitions


def eval_rollout(
    rng: jax.Array,
    state: hk.State,
    env: Environment,
    config: dict,
    ts_policy: TrainState,
    action_dim: int,
    steps_per_rollout: int,
) -> RolloutStats:

    stats = RolloutStats(episode_return=0, length=0)

    rng, reset_rng = jax.random.split(rng, 2)
    xtimestep = env.reset(env.env_params, reset_rng)
    task = xtimestep.timestep.state.goal
    if len(task.shape) == 1:
        task = task[jnp.newaxis]

    prev_action = jnp.zeros((1, action_dim))
    prev_reward = jnp.zeros((1, 1))
    done = False

    def _step_fn(carry, _):
        (
            rng,
            stats,
            state,
            xtimestep,
            prev_action,
            prev_reward,
            done,
        ) = carry

        rng, policy_rng, encoder_rng = jax.random.split(rng, 3)
        observation = xtimestep.timestep.observation
        observation = observation.astype(jnp.float32)
        observation = observation[jnp.newaxis]

        policy_output, state = ts_policy.apply_fn(
            ts_policy.params,
            state,
            policy_rng,
            env_state=observation,
            task=task,
            is_training=False,
        )
        action = policy_output.action
        xtimestep = env.step(env.env_params, xtimestep, action.squeeze())
        timestep = xtimestep.timestep
        next_obs = timestep.observation
        reward = timestep.reward
        done = timestep.last()
        next_obs = next_obs.astype(jnp.float32)

        # add extra dimension for batch
        next_obs = next_obs[jnp.newaxis]
        action = action.reshape(1, 1).astype(jnp.float32)
        reward = reward.reshape(1, 1)

        stats = stats.replace(
            episode_return=stats.episode_return + timestep.reward,
            length=stats.length + 1,
        )

        return (
            rng,
            stats,
            state,
            xtimestep,
            action,
            reward,
            done,
        ), (timestep, action.squeeze(axis=-1))

    init_carry = (
        rng,
        stats,
        state,
        xtimestep,
        prev_action,
        prev_reward,
        done,
    )

    # transitions contains (timestep, action)
    carry, transitions = jax.lax.scan(
        _step_fn, init_carry, None, length=steps_per_rollout
    )
    stats = carry[1]
    return stats, transitions


def eval_rollout_with_belief_model(
    rng: jax.Array,
    state: hk.State,
    env: Environment,
    config: dict,
    ts_policy: TrainState,
    ts_vae: TrainState,
    get_prior: Callable,
    action_dim: int,
    steps_per_rollout: int,
) -> RolloutStats:

    stats = RolloutStats(episode_return=0, length=0)
    vae_state, policy_state = state

    rng, reset_rng, prior_rng = jax.random.split(rng, 3)
    xtimestep = env.reset(env.env_params, reset_rng)
    observation = xtimestep.timestep.observation
    prev_action = jnp.zeros((1, action_dim))
    prev_reward = jnp.zeros((1, 1))
    done = False

    # if using a transformer, we need to keep track of the full history
    if config.vae.encoder.name == "transformer":
        prev_states = np.zeros((steps_per_rollout, 1, *observation.shape))
        prev_action = np.zeros((steps_per_rollout, 1, action_dim))
        prev_reward = np.zeros((steps_per_rollout, 1, 1))
        masks = np.zeros((steps_per_rollout, 1))

    prior_outputs = get_prior(ts_vae.params, prior_rng, batch_size=1)
    latent_mean = prior_outputs.latent_mean
    latent_logvar = prior_outputs.latent_logvar
    latent = jnp.concatenate([latent_mean, latent_logvar], axis=-1)
    hidden_state = prior_outputs.hidden_state

    def _step_fn(carry, _):
        (
            rng,
            stats,
            vae_state,
            policy_state,
            xtimestep,
            prev_action,
            prev_reward,
            done,
            latent,
            hidden_state,
        ) = carry

        rng, policy_rng, encoder_rng = jax.random.split(rng, 3)
        observation = xtimestep.timestep.observation
        observation = observation.astype(jnp.float32)
        observation = observation[jnp.newaxis]

        policy_output, policy_state = ts_policy.apply_fn(
            ts_policy.params,
            policy_state,
            policy_rng,
            env_state=observation,
            latent=latent,
            is_training=False,
        )
        action = policy_output.action
        xtimestep = env.step(env.env_params, xtimestep, action.squeeze())
        timestep = xtimestep.timestep
        next_obs = timestep.observation
        reward = timestep.reward
        done = timestep.last()
        next_obs = next_obs.astype(jnp.float32)

        # add extra dimension for batch
        next_obs = next_obs[jnp.newaxis]
        action = action.reshape(1, 1).astype(jnp.float32)
        reward = reward.reshape(1, 1)

        if config.vae.encoder.name == "lstm":
            # update hidden state
            encode_outputs, vae_state = ts_vae.apply_fn(
                ts_vae.params,
                vae_state,
                encoder_rng,
                states=next_obs,
                actions=action,
                rewards=reward,
                hidden_state=hidden_state,
                is_training=False,
            )
        elif config.vae.encoder.name == "transformer":
            pass
            # states[step] = next_state
            # actions[step] = action
            # rewards[step] = reward
            # masks[step] = 1.0 - done.flatten()

            # # we want to embed the full sequence cause jax
            # encode_outputs, self.vae_state = self.ts_vae.apply_fn(
            #     self.ts_vae.params,
            #     self.vae_state,
            #     next(self.rng_seq),
            #     states=states,
            #     actions=actions,
            #     rewards=rewards,
            #     mask=masks,
            #     is_training=True,
            # )

            # # take the last timestep for every item
            # encode_outputs.latent_mean = encode_outputs.latent_mean[step]
            # encode_outputs.latent_logvar = encode_outputs.latent_logvar[step]
            # encode_outputs.latent_sample = encode_outputs.latent_sample[step]

        hidden_state = encode_outputs.hidden_state
        latent_mean = encode_outputs.latent_mean
        latent_logvar = encode_outputs.latent_logvar
        latent = jnp.concatenate([latent_mean, latent_logvar], axis=-1)

        stats = stats.replace(
            episode_return=stats.episode_return + timestep.reward,
            length=stats.length + 1,
        )

        return (
            rng,
            stats,
            vae_state,
            policy_state,
            xtimestep,
            action,
            reward,
            done,
            latent,
            hidden_state,
        ), (timestep, action.squeeze(axis=-1))

    init_carry = (
        rng,
        stats,
        vae_state,
        policy_state,
        xtimestep,
        prev_action,
        prev_reward,
        done,
        latent,
        hidden_state,
    )

    # transitions contains (timestep, action)
    carry, transitions = jax.lax.scan(
        _step_fn, init_carry, None, length=steps_per_rollout
    )
    stats = carry[1]
    return stats, transitions
