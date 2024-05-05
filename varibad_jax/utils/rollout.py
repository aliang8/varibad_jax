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
import matplotlib.pyplot as plt
from xminigrid.experimental.img_obs import TILE_W_AGENT_CACHE, TILE_CACHE, TILE_SIZE
from xminigrid.core.constants import NUM_COLORS, TILES_REGISTRY, Tiles


@chex.dataclass
class RolloutStats:
    episode_return: jnp.ndarray
    length: jnp.ndarray


def run_rollouts(
    rng: jax.random.PRNGKey,
    agent,
    config: ConfigDict,
    env: Environment,
    steps_per_rollout: int,
    action_dim: int,
    wandb_run=None,
    prompts=None,
    **kwargs,
) -> RolloutStats:
    logging.info("Evaluating policy rollout...")

    rng_keys = jax.random.split(rng, config.num_eval_rollouts)

    start = time.time()

    rollout_kwargs = dict(
        agent=agent,
        action_dim=action_dim,
        steps_per_rollout=steps_per_rollout,
        env=env,
        config=config,
        **kwargs,
    )

    if config.trainer == "vae":
        rollout_fn = eval_rollout_with_belief_model
    elif "dt" in config.model.name:
        rollout_fn = eval_rollout_dt
    else:
        rollout_fn = eval_rollout

    # render function doesn't work with vmap
    if prompts is not None:
        logging.info("using prompts")
        eval_metrics, (transitions, actions) = jax.vmap(
            lambda rng, prompt: functools.partial(rollout_fn, **rollout_kwargs)(
                rng=rng, prompt=prompt
            )
        )(rng_keys, prompts)
    else:
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

        if config.env.env_name == "gridworld":
            trajectories = transitions.observation
            goals = transitions.state.goal

            fig, axes = plt.subplots(figsize=(12, 12), nrows=3, ncols=3)
            grid_size = env.env_params.grid_size

            for traj_indx, ax in enumerate(axes.flatten()):
                for i in range(grid_size):
                    for j in range(grid_size):
                        # add square with border
                        ax.add_patch(
                            plt.Rectangle((i, j), 1, 1, fill=None, edgecolor="black")
                        )

                trajectory = trajectories[traj_indx]
                start_location = trajectory[0]
                ax.plot(start_location[0] + 0.5, start_location[1] + 0.5, "ro")

                # show trajectory with a blue line
                # remove the last point because it just got reset
                for i in range(len(trajectory) - 2):
                    start = trajectory[i]
                    end = trajectory[i + 1]
                    ax.plot(
                        [start[0] + 0.5, end[0] + 0.5],
                        [start[1] + 0.5, end[1] + 0.5],
                        color=(0, i / len(trajectory), 0),
                        linewidth=3,
                    )

                # mark goal with a green
                goal_location = goals[traj_indx][0]
                ax.plot(
                    goal_location[0] + 0.5, goal_location[1] + 0.5, "gx", markersize=10
                )
                ax.axis("off")

                # label side of grid with numbers
                for i in range(grid_size):
                    ax.text(i + 0.5, -0.5, str(i), ha="center", va="center")
                    ax.text(-0.5, i + 0.5, str(i), ha="center", va="center")

            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            # add the image to wandb
            wandb_run.log({f"eval_rollouts/trajectory": wandb.Image(fig)})

        else:
            # imgs is N x T x H x W x C
            # we want N x T x C x H x W

            # generate the images
            videos = []
            for rollout_indx in range(config.num_eval_rollouts):
                images = []
                for step in range(steps_per_rollout):
                    timestep = jtu.tree_map(
                        lambda x: x[rollout_indx][step], transitions
                    )
                    images.append(env.render(env.env_params, timestep))
                videos.append(images)

            videos = np.array(videos)

            videos = einops.rearrange(videos, "n t h w c -> n t c h w")
            wandb_run.log({"eval_rollouts": wandb.Video(np.array(videos), fps=5)})

    return eval_metrics, transitions, actions


def eval_rollout_dt(
    rng: jax.Array,
    agent,
    env: Environment,
    config: dict,
    action_dim: int,
    steps_per_rollout: int,
    prompt=None,
    **kwargs,
) -> RolloutStats:
    rng, reset_rng = jax.random.split(rng, 2)
    if config.model.policy.demo_conditioning and prompt is not None:
        # set the goal to be the same as the one in the prompt
        desired_goal = prompt["tasks"][0]

    xtimestep = env.reset(env.env_params, reset_rng, desired_goal=desired_goal)
    observation_shape = xtimestep.timestep.observation.shape

    # first dimension is batch
    observations = jnp.zeros((1, steps_per_rollout, *observation_shape))
    actions = jnp.zeros((1, steps_per_rollout, action_dim))
    rewards = jnp.zeros((1, steps_per_rollout, 1))
    mask = jnp.zeros((1, steps_per_rollout))
    done = False
    start_index = 0

    if config.model.policy.task_conditioning:
        if config.env.env_name == "gridworld":
            task = xtimestep.timestep.state.goal
        elif config.env.env_name == "xland":
            task = xtimestep.timestep.state.goal_encoding

        task = task.reshape(1, 1, -1)
        prompt = task
    elif config.model.policy.demo_conditioning:
        logging.info("adding prompt info before trajectory")
        # prepend prompt into the input tokens
        num_prompt_steps = prompt["observations"].shape[0]
        observations = jnp.concatenate(
            [prompt["observations"][jnp.newaxis], observations], axis=1
        )
        if "lam" in config.model.name:
            rng, latent_rng = jax.random.split(rng)
            # relabel demo trajectory with latent actions
            latent_actions = agent.label_trajectory_with_actions(
                latent_rng, prompt["observations"][jnp.newaxis]
            )
            actions = jnp.concatenate([latent_actions, actions], axis=1)
        else:
            actions = jnp.concatenate([prompt["actions"][jnp.newaxis], actions], axis=1)
        rewards = jnp.concatenate(
            [prompt["rewards"].reshape(1, -1, 1), rewards], axis=1
        )

        mask = jnp.concatenate([jnp.ones((1, num_prompt_steps)), mask], axis=1)
        start_index = num_prompt_steps
        prompt = None
    else:
        prompt = None

    # jax.debug.breakpoint()

    stats = RolloutStats(episode_return=0, length=start_index)

    # first return-to-go is the desired return value
    # this is probably environment / task specific
    rtg = 60.0
    rewards = rewards.at[0, start_index].set(rtg)

    def _step_fn(carry, _):
        (
            rng,
            stats,
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

        policy_output, _ = agent.get_action(
            policy_rng,
            env_state=None,
            states=observations,
            actions=actions,
            rewards=rewards if config.model.policy.use_rtg else None,
            mask=mask,
            prompt=prompt,
            is_training=False,
        )

        entropy = policy_output.entropy
        entropy = jnp.mean(entropy)
        logging.info(f"entropy: {entropy}")

        # [B, T]
        action = policy_output.action
        action = action[0, stats.length]
        xtimestep = env.step(env.env_params, xtimestep, action)
        timestep = xtimestep.timestep
        reward = timestep.reward
        done = timestep.last()

        action = action.reshape(1).astype(jnp.float32)
        reward = reward.reshape(1)

        if "lam" in config.model.name:
            action = policy_output.latent_action[0, stats.length]

        actions = actions.at[0, stats.length].set(action)

        # return to go is the previous return minus rewad
        rtg = rtg - timestep.reward

        rewards = rewards.at[0, stats.length].set(rtg)

        stats = stats.replace(
            episode_return=stats.episode_return + timestep.reward,
            length=stats.length + 1,
        )

        if action.shape[-1] == 1:
            action = action.squeeze(axis=-1)

        return (
            rng,
            stats,
            xtimestep,
            observations,
            actions,
            rewards,
            rtg,
            mask,
            done,
        ), (timestep, action)

    init_carry = (
        rng,
        stats,
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
    agent,
    env: Environment,
    config: dict,
    action_dim: int,
    steps_per_rollout: int,
    prompt=None,
    **kwargs,
) -> RolloutStats:

    stats = RolloutStats(episode_return=0, length=0)

    rng, reset_rng = jax.random.split(rng, 2)

    if prompt is not None:
        # set the goal to be the same as the one in the prompt
        # TODO: fix this, might not be correct shape
        desired_goal = prompt["tasks"]
        xtimestep = env.reset(env.env_params, reset_rng, desired_goal=desired_goal)
    else:
        xtimestep = env.reset(env.env_params, reset_rng)

    if config.env.env_name == "gridworld":
        task = xtimestep.timestep.state.goal
    elif config.env.env_name == "xland":
        task = xtimestep.timestep.state.goal_encoding

    if len(task.shape) == 1:
        task = task[jnp.newaxis]

    task = task.astype(jnp.float32)

    prev_action = jnp.zeros((1, action_dim))
    prev_reward = jnp.zeros((1, 1))
    done = False

    def _step_fn(carry, _):
        (
            rng,
            stats,
            xtimestep,
            prev_action,
            prev_reward,
            done,
        ) = carry

        rng, policy_rng = jax.random.split(rng, 2)
        observation = xtimestep.timestep.observation
        observation = observation.astype(jnp.float32)
        observation = observation[jnp.newaxis]

        if config.env.env_name == "gridworld":
            task = xtimestep.timestep.state.goal
        elif config.env.env_name == "xland":
            task = xtimestep.timestep.state.goal_encoding

        if len(task.shape) == 1:
            task = task[jnp.newaxis]

        task = task.astype(jnp.float32)

        logging.info(
            f"observation shape: {observation.shape}, task shape: {task.shape}"
        )

        policy_output, _ = agent.get_action(
            policy_rng,
            env_state=observation,
            task=task,
            is_training=False,
        )
        action = policy_output.action
        next_xtimestep = env.step(env.env_params, xtimestep, action.squeeze())
        next_timestep = next_xtimestep.timestep
        next_obs = next_timestep.observation
        reward = next_timestep.reward
        done = next_timestep.last()
        next_obs = next_obs.astype(jnp.float32)

        # add extra dimension for batch
        next_obs = next_obs[jnp.newaxis]
        action = action.reshape(1, 1).astype(jnp.float32)
        reward = reward.reshape(1, 1)

        if "lam" in config.model.name:
            action = policy_output.latent_action

        stats = stats.replace(
            episode_return=stats.episode_return + next_timestep.reward,
            length=stats.length + 1,
        )

        if action.shape[-1] == 1:
            action = action.squeeze(axis=-1)

        return (
            rng,
            stats,
            next_xtimestep,
            action,
            reward,
            done,
        ), (xtimestep.timestep, action)

    init_carry = (
        rng,
        stats,
        xtimestep,
        prev_action,
        prev_reward,
        done,
    )

    # transitions contains (timestep, action)
    carry, (transitions, actions) = jax.lax.scan(
        _step_fn, init_carry, None, length=steps_per_rollout
    )
    stats = carry[1]
    return stats, (transitions, actions)


def eval_rollout_with_belief_model(
    rng: jax.Array,
    agent,
    belief_model,
    env: Environment,
    config: dict,
    action_dim: int,
    steps_per_rollout: int,
) -> RolloutStats:

    stats = RolloutStats(episode_return=0, length=0)

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

    prior_outputs = belief_model.get_prior(prior_rng, batch_size=1)
    latent_mean = prior_outputs.latent_mean
    latent_logvar = prior_outputs.latent_logvar
    latent = jnp.concatenate([latent_mean, latent_logvar], axis=-1)
    hidden_state = prior_outputs.hidden_state

    def _step_fn(carry, _):
        (
            rng,
            stats,
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

        policy_output, policy_state = agent.get_action(
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
            encode_outputs, _ = belief_model.encode_trajectory(
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
