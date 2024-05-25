import time
from absl import logging
import jax
import chex
import tqdm
import haiku as hk
import jax.numpy as jnp
from xminigrid.environment import Environment, EnvParamsT
from flax.training.train_state import TrainState
from typing import Callable, Union, Tuple
import numpy as np
import wandb
import cv2
import einops
import functools
from ml_collections import ConfigDict
import jax.tree_util as jtu
import matplotlib.pyplot as plt

from varibad_jax.utils.data_utils import subsample_data
from xminigrid.experimental.img_obs import TILE_W_AGENT_CACHE, TILE_CACHE, TILE_SIZE
from xminigrid.core.constants import NUM_COLORS, TILES_REGISTRY, Tiles, Colors
from xminigrid.types import AgentState, EnvCarry, GridState, RuleSet, State


@chex.dataclass
class RolloutStats:
    episode_return: jnp.ndarray
    length: jnp.ndarray
    success: jnp.ndarray


@chex.dataclass
class Transition:
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


def run_rollouts_procgen(rng, agent, config: ConfigDict, env, wandb_run=None):
    logging.info("Rollout procgen...")
    obs = env.reset()

    rng, policy_rng = jax.random.split(rng, 2)

    dones = jnp.zeros((config.num_eval_rollouts,))
    ep_returns = jnp.zeros((config.num_eval_rollouts,))
    ep_lengths = jnp.zeros((config.num_eval_rollouts,))
    transitions = []

    rollout_time = time.time()

    while not jnp.all(dones):
        obs = obs.astype(jnp.float32)
        (policy_output, _), _ = agent.get_action(
            policy_rng, env_state=obs, is_training=False
        )
        action = policy_output.action
        obs, reward, done, _ = env.step(action)

        transition = Transition(
            observation=obs,
            action=action,
            reward=reward,
            done=done,
        )

        transitions.append(transition)

        for i in range(config.num_eval_rollouts):
            if not done[i]:
                ep_returns = ep_returns.at[i].set(ep_returns[i] + reward[i])
                ep_lengths = ep_lengths.at[i].set(ep_lengths[i] + 1)

        # if wandb_run is not None:
        #     wandb_run.log({"procgen_rollouts": wandb.Video(np.array(obs), fps=5)})

    rollout_time = time.time() - rollout_time

    eval_metrics = {
        "episode_return": jnp.mean(ep_returns),
        "avg_length": jnp.mean(ep_lengths),
        "rollout_time": rollout_time / config.num_eval_rollouts,
    }
    return eval_metrics, transitions


def run_rollouts(
    rng: jax.random.PRNGKey,
    agent,
    config: ConfigDict,
    env: Environment,
    env_id: str,
    action_dim: int,
    wandb_run=None,
    prompts=None,
    **kwargs,
) -> RolloutStats:
    logging.info("Evaluating policy rollout...")

    start = time.time()

    rollout_kwargs = dict(
        rng=rng,
        agent=agent,
        action_dim=action_dim,
        env=env,
        config=config,
        **kwargs,
    )

    if config.trainer == "vae":
        rollout_fn = eval_rollout_with_belief_model
    elif "dt" in config.model.name or "icl" in config.model.name:
        rollout_fn = eval_rollout_dt
        rollout_kwargs["max_traj_len"] = 100
    else:
        rollout_fn = eval_rollout

    # render function doesn't work with vmap
    if prompts is not None:
        logging.info("using prompts")
        eval_metrics, (trajectories, actions, successes) = rollout_fn(
            prompts=prompts, **rollout_kwargs
        )
    else:
        eval_metrics, (trajectories, actions, successes) = rollout_fn(**rollout_kwargs)

    rollout_time = time.time() - start
    eval_metrics = {
        "episode_return": eval_metrics["episode_return"],
        "avg_length": eval_metrics["length"],
        "success": eval_metrics["success"],
        "rollout_time": rollout_time / config.num_eval_rollouts,
    }

    # visualize the rollouts
    if wandb_run is not None and config.visualize_rollouts:

        if config.env.env_name == "gridworld":
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
            num_render = min(config.num_eval_rollouts_render, len(trajectories))
            for rollout_indx in range(num_render):
                images = []
                for step, timestep in enumerate(trajectories[rollout_indx]):
                    img = env.render(env.env_params, timestep)
                    # cv2 text image
                    img = cv2.putText(
                        img,
                        f"step: {step}, s: {successes[rollout_indx]}",
                        (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )
                    images.append(img)
                videos.append(images)

            # pad videos to same length
            max_length = max([len(video) for video in videos])
            for indx, video in enumerate(videos):
                while len(video) < max_length:
                    video.append(video[-1])

                videos[indx] = video

            videos = np.array(videos)
            videos = einops.rearrange(videos, "n t h w c -> n t c h w")
            wandb_run.log(
                {f"{env_id}/eval_rollouts": wandb.Video(np.array(videos), fps=5)}
            )

    return eval_metrics, (trajectories, actions, successes)


def eval_rollout_dt(
    rng: jax.Array,
    agent,
    env: Environment,
    config: dict,
    action_dim: int,
    max_traj_len: int,
    prompts=None,
    **kwargs,
) -> RolloutStats:
    jit_reset = jax.jit(env.reset, static_argnames=("randomize_agent"))
    jit_step = jax.jit(env.step)
    jit_apply = jax.jit(agent.get_action, static_argnames=("is_training"))

    if "lam" in config.model.name:
        jit_relabel = jax.jit(agent.label_trajectory_with_actions)

    trajectories = []
    all_actions = []
    all_stats = []
    all_successes = []

    for episode in tqdm.tqdm(range(config.num_eval_rollouts)):
        rng, reset_rng = jax.random.split(rng, 2)

        prompt = subsample_data(prompts, episode)

        timestep = env_reset(config, reset_rng, env, jit_reset, prompt=prompt)
        timesteps = [timestep]
        traj_actions = []

        if config.env.env_name == "gridworld":
            task = timestep.state.goal
        elif config.env.env_name == "xland":
            task = timestep.state.goal_encoding

        if len(task.shape) == 1:
            task = task[jnp.newaxis]

        observation_shape = timestep.observation.shape

        # first dimension is batch
        observations = np.zeros((1, max_traj_len, *observation_shape))
        actions = np.zeros((1, max_traj_len, action_dim))
        rewards = np.zeros((1, max_traj_len, 1))
        mask = np.zeros((1, max_traj_len))
        start_index = 0

        if config.model.policy.task_conditioning:
            if config.env.env_name == "gridworld":
                task = timestep.state.goal
            elif config.env.env_name == "xland":
                task = timestep.state.goal_encoding

            task = task.reshape(1, 1, -1)
            prompt = task
        elif (
            config.model.policy.demo_conditioning
            and config.data.num_trajs_per_batch > 1
        ):
            # logging.info("adding prompt info before trajectory")
            # prepend prompt into the input tokens
            prompt_steps = int(prompt["mask"].sum())

            # filter prompt for steps
            filtered_prompt = jtu.tree_map(lambda x: x[:prompt_steps], prompt)
            observations[:, :prompt_steps] = filtered_prompt["observations"]

            if "lam" in config.model.name:
                rng, latent_rng = jax.random.split(rng)
                # relabel demo trajectory with latent actions
                latent_actions = jit_relabel(
                    latent_rng, prompt["observations"][jnp.newaxis].astype(jnp.float32)
                )
                # TODO: might be an index issue here
                latent_actions = latent_actions[:, :prompt_steps]
                actions[:, :prompt_steps] = latent_actions
            else:
                actions[:, :prompt_steps] = filtered_prompt["actions"]

            rewards[:, :prompt_steps] = filtered_prompt["rewards"].reshape(1, -1, 1)
            mask[:, :prompt_steps] = 1.0
            traj_index = np.zeros((1, max_traj_len))
            traj_index[:, :prompt_steps] = filtered_prompt["traj_index"][:prompt_steps]
            traj_index = jnp.array(traj_index)

            start_index = prompt_steps
            prompt = None
        else:
            prompt = None
            traj_index = None

        # first return-to-go is the desired return value
        # this is probably environment / task specific
        rtg = 60.0
        observations = jnp.array(observations)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        mask = jnp.array(mask)
        rewards = rewards.at[0, start_index].set(rtg)

        step = 0
        success = False
        done = False
        stats = RolloutStats(episode_return=0, length=start_index, success=False)

        while not done:
            rng, policy_rng = jax.random.split(rng)
            observation = timestep.observation
            observation = observation.astype(jnp.float32)
            observations = observations.at[0, stats.length].set(observation)
            mask = mask.at[0, stats.length].set(1.0)

            if (
                config.model.policy.demo_conditioning
                and config.data.num_trajs_per_batch > 1
            ):
                traj_index = traj_index.at[0, stats.length].set(
                    config.data.num_trajs_per_batch - 1
                )

            policy_output, _ = jit_apply(
                policy_rng,
                env_state=None,
                states=observations,
                actions=actions,
                rewards=rewards if config.model.policy.use_rtg else None,
                mask=mask,
                prompt=prompt,
                traj_index=traj_index,
                is_training=False,
            )

            entropy = policy_output.entropy
            entropy = jnp.mean(entropy)
            # logging.info(f"entropy: {entropy}")

            # [B, T]
            action = policy_output.action
            action = action[0, stats.length]
            timestep = jit_step(env.env_params, timestep, action)
            reward = timestep.reward
            done = timestep.last()

            action = action.reshape(1).astype(jnp.float32)
            reward = reward.reshape(1)

            if step < (env.max_episode_steps - 1):
                success = success or done

            if "lam" in config.model.name:
                action = policy_output.latent_action[0, stats.length]

            actions = actions.at[0, stats.length].set(action)

            # return to go is the previous return minus reward
            rtg = rtg - timestep.reward

            rewards = rewards.at[0, stats.length].set(rtg)

            if action.shape[-1] == 1:
                action = action.squeeze(axis=-1)

            step += 1
            timesteps.append(timestep)
            traj_actions.append(action)

            stats = stats.replace(
                episode_return=stats.episode_return + timestep.reward,
                length=stats.length + 1,
                success=success,
            )

        trajectories.append(timesteps)
        all_actions.append(traj_actions)
        all_stats.append(stats)
        all_successes.append(success)
        all_transitions = trajectories

    stats = jtu.tree_map(lambda *x: jnp.mean(jnp.stack(x)), *all_stats)
    return stats, (all_transitions, all_actions, all_successes)


def env_reset(config: ConfigDict, rng, env, reset_fn: Callable, prompt: dict = None):

    rng, reset_rng = jax.random.split(rng, 2)

    if prompt is not None:
        # set the goal to be the same as the one in the prompt

        if config.env.env_name == "gridworld":
            state = prompt["tasks"][0]
        elif config.env.env_name == "xland":
            # reset the state of the environment to be the same as the prompt
            # agent position and dir will be randomized still
            agent_position = prompt["info"]["agent_position"][0].astype(jnp.int32)
            agent_direction = prompt["info"]["agent_direction"][0].astype(jnp.int32)
            grid = prompt["info"]["grid"][0].astype(jnp.uint8)

            # # find out where the goal is
            # goal_tile = TILES_REGISTRY[Tiles.BALL, Colors.YELLOW]

            # jax.debug.breakpoint()
            # goal_locs = jnp.array(
            #     list(zip(*jnp.where((grid == goal_tile).sum(axis=-1))))
            # )
            # new_tile = TILES_REGISTRY[Tiles.BALL, Colors.RED]
            # jax.debug.breakpoint()

            goal_encoding = prompt["info"]["goal"][0].astype(jnp.uint8)
            rule_encoding = prompt["info"]["rule"][0].astype(jnp.uint8)
            agent_state = AgentState(position=agent_position, direction=agent_direction)
            state = State(
                key=rng,
                step_num=jnp.asarray(0),
                grid=grid,
                agent=agent_state,
                goal_encoding=goal_encoding,
                rule_encoding=rule_encoding,
                carry=EnvCarry(),
            )

        # import ipdb

        # ipdb.set_trace()
        timestep = reset_fn(
            env.env_params, reset_rng, state=state, randomize_agent=True
        )
    else:
        timestep = reset_fn(env.env_params, reset_rng)

    return timestep


def eval_rollout(
    rng: jax.Array,
    agent,
    env: Environment,
    config: dict,
    prompt=None,
    **kwargs,
) -> RolloutStats:

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_apply = jax.jit(agent.get_action, static_argnames=("is_training"))

    trajectories = []
    all_actions = []
    all_stats = []
    all_successes = []

    for episode in tqdm.tqdm(range(config.num_eval_rollouts)):
        stats = RolloutStats(episode_return=0, length=0, success=False)
        rng, reset_rng = jax.random.split(rng, 2)

        timestep = env_reset(config, reset_rng, env, jit_reset, prompt=prompt)

        timesteps = [timestep]
        actions = []

        if config.env.env_name == "gridworld":
            task = timestep.state.goal
        elif config.env.env_name == "xland":
            task = timestep.state.goal_encoding

        if len(task.shape) == 1:
            task = task[jnp.newaxis]

        task = task.astype(jnp.float32)

        success = False
        done = False

        if config.model.policy.use_rnn_policy:
            hidden_state = jnp.zeros((1, config.model.policy.rnn_hidden_size))
        else:
            hidden_state = None

        step = 0

        while not done:
            rng, policy_rng = jax.random.split(rng, 2)
            observation = timestep.observation
            observation = observation.astype(jnp.float32)
            observation = observation[jnp.newaxis]

            if config.env.env_name == "gridworld":
                task = timestep.state.goal
            elif config.env.env_name == "xland":
                task = timestep.state.goal_encoding

            if len(task.shape) == 1:
                task = task[jnp.newaxis]

            task = task.astype(jnp.float32)

            (policy_output, hidden_state), _ = jit_apply(
                policy_rng,
                env_state=observation,
                hidden_state=hidden_state,
                task=task,
                is_training=False,
            )

            action = policy_output.action
            next_timestep = jit_step(env.env_params, timestep, action.squeeze())
            next_obs = next_timestep.observation
            reward = next_timestep.reward
            done = next_timestep.last()

            if step < (env.max_episode_steps - 1):
                success = success or done

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
                success=success,
            )

            timesteps.append(next_timestep)
            actions.append(action)
            step += 1
            timestep = next_timestep

        # add a dummy action for the shape
        actions.append(actions[-1])

        # only add successful trajectories
        # if success:
        trajectories.append(timesteps)
        all_actions.append(actions)
        all_stats.append(stats)
        all_successes.append(success)

    # merge all the trajectories
    # all_transitions = [
    #     transition for trajectory in trajectories for transition in trajectory
    # ]

    # # combine transitions
    # all_transitions = jtu.tree_map(lambda *v: jnp.stack(v), *all_transitions)

    # # merge actions too
    # all_actions = [action for actions in all_actions for action in actions]
    # all_actions = jtu.tree_map(lambda *v: jnp.stack(v), *all_actions)
    # all_successes = jnp.array(all_successes)

    all_transitions = trajectories

    stats = jtu.tree_map(lambda *x: jnp.mean(jnp.stack(x)), *all_stats)

    # if action.shape[-1] == 1:
    #     action = action.squeeze(axis=-1)
    return stats, (all_transitions, all_actions, all_successes)


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
    timestep = env.reset(env.env_params, reset_rng)
    observation = timestep.observation
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
            timestep,
            prev_action,
            prev_reward,
            done,
            latent,
            hidden_state,
        ) = carry

        rng, policy_rng, encoder_rng = jax.random.split(rng, 3)
        observation = timestep.observation
        observation = observation.astype(jnp.float32)
        observation = observation[jnp.newaxis]

        policy_output, policy_state = agent.get_action(
            policy_rng,
            env_state=observation,
            latent=latent,
            is_training=False,
        )
        action = policy_output.action
        timestep = env.step(env.env_params, timestep, action.squeeze())
        timestep = timestep
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
            timestep,
            action,
            reward,
            done,
            latent,
            hidden_state,
        ), (timestep, action.squeeze(axis=-1))

    init_carry = (
        rng,
        stats,
        timestep,
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
