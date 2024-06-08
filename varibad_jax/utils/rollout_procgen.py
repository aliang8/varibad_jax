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
import seaborn as sns
from matplotlib import colors
from typing import Dict

from varibad_jax.envs.utils import make_envs, make_procgen_envs
from varibad_jax.utils.data_utils import subsample_data, normalize_obs


@chex.dataclass
class Transition:
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


def run_rollouts_procgen(
    rng: jax.random.PRNGKey,
    agent,
    config: ConfigDict,
    env: Environment,
    env_id: str,
    action_dim: int,
    wandb_run=None,
    prompts=None,
    **kwargs,
):
    rollout_kwargs = dict(
        rng=rng,
        agent=agent,
        action_dim=action_dim,
        env=env,
        env_id=env_id,
        config=config,
        **kwargs,
    )

    if "dt" in config.model.name or "icl" in config.model.name:
        rollout_fn = run_rollouts_procgen_dt
    else:
        rollout_fn = run_rollouts_procgen_helper

    if prompts is not None:
        logging.info("using prompts")
        eval_metrics, transitions = rollout_fn(prompts=prompts, **rollout_kwargs)
    else:
        eval_metrics, transitions = rollout_fn(**rollout_kwargs)

    # flatten list of transitions
    transitions = jtu.tree_map(lambda *x: np.stack(x), *transitions)
    # swap first two axes
    transitions = jtu.tree_map(lambda x: np.swapaxes(x, 0, 1), transitions)

    if wandb_run is not None:
        videos = transitions.observation
        videos = videos[: config.num_eval_rollouts_render]

        videos_np = []
        for rollout_indx, video in enumerate(videos):

            tmp = []
            for step, frame in enumerate(video):
                returns = transitions.reward[rollout_indx][:step].sum().item()
                action = transitions.action[rollout_indx][step].item()
                done = transitions.done[rollout_indx][step].item()
                # ep_len = ep_lengths[rollout_indx].item()
                frame = cv2.putText(
                    np.array(frame),
                    "",
                    # f"t: {step}, len: {ep_len}, a = {action}, r = {returns}, d = {done}",
                    (5, 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5e-2,
                    (255, 255, 255),
                    1,
                )
                tmp.append(frame)
            videos_np.append(tmp)

        videos = np.array(videos_np)
        videos = einops.rearrange(videos, "n t h w c -> n t c h w")
        wandb_run.log({f"{env_id}/eval_rollouts": wandb.Video(np.array(videos), fps=5)})

    return eval_metrics, transitions


def run_rollouts_procgen_dt(
    rng,
    agent,
    config: ConfigDict,
    env,
    env_id: str,
    action_dim: int,
    prompts: Dict = None,
    **kwargs,
):
    # gym version needs to be gym==0.23.1 for this to work
    logging.info("rollout procgen dt...")
    jit_apply = jax.jit(agent.get_action, static_argnames=("is_training"))
    if "lam" in config.model.name:
        jit_relabel = jax.jit(agent.label_trajectory_with_actions)

    # max_traj_len = (
    #     config.data.context_window if config.data.context_window > 1 else 1000
    # )
    max_traj_len = 1000
    rollout_time = time.time()

    ep_returns = np.zeros((config.num_eval_rollouts,))
    ep_lengths = np.zeros((config.num_eval_rollouts,))

    # doing linear rollouts for now, but will batch it later (TODO)
    for episode in tqdm.tqdm(range(config.num_eval_rollouts)):
        config.env.num_envs = 1
        env = make_procgen_envs(training=False, **config.env)
        obs = env.reset()

        prompt = subsample_data(prompts, episode)

        # first dimension is batch
        observation_shape = obs.shape[1:]
        observations = np.zeros((1, max_traj_len, *observation_shape))
        actions = np.zeros((1, max_traj_len, action_dim))
        rewards = np.zeros((1, max_traj_len, 1))
        mask = np.zeros((1, max_traj_len))
        dt_timestep = np.arange(max_traj_len).reshape(1, -1)
        # dones = np.zeros((1,))

        start_index = 0

        if config.model.policy.task_conditioning:
            import ipdb

            ipdb.set_trace()
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

            # sample a couple steps from the prompt
            si = np.random.randint(0, prompt_steps - config.data.context_window)
            filtered_prompt = jtu.tree_map(
                lambda x: x[si : si + config.data.context_window], filtered_prompt
            )

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

            dt_timestep = np.zeros((1, max_traj_len))
            dt_timestep[:, :prompt_steps] = filtered_prompt["timestep"][:prompt_steps]
            dt_timestep = jnp.array(dt_timestep)

            start_index = prompt_steps
            prompt = None
        else:
            prompt = None
            traj_index = None
            # dt_timestep = None

        curr_index = start_index
        done_ = False

        # set the first observation and mask
        observations[0, start_index] = obs
        mask[0, start_index] = 1.0

        transitions = []

        while not done_:
            # print(curr_index, done_)
            # break when either the episode terminates or we reach max timesteps
            done_ = jnp.logical_or(
                done_, ep_lengths[episode] >= config.env.steps_per_rollout
            )
            if done_:
                break
            # if done.all():
            #     break

            rng, policy_rng = jax.random.split(rng)

            # get context window of K steps, check the indexing (TODO)
            si = max(0, curr_index - config.data.context_window)
            ei = max(curr_index, config.data.context_window)
            print(si, ei)
            obs_context = observations[:, si:ei]
            obs_context = normalize_obs(obs_context.astype(jnp.float32))
            action_context = actions[:, si:ei]
            reward_context = rewards[:, si:ei] if config.model.policy.use_rtg else None
            mask_context = mask[:, si:ei]
            dt_timestep_context = dt_timestep[:, si:ei]

            policy_output, _ = jit_apply(
                policy_rng,
                env_state=None,
                states=obs_context,
                actions=action_context,
                rewards=reward_context,
                mask=mask_context,
                timestep=dt_timestep_context,
                prompt=prompt,
                traj_index=traj_index,
                is_training=False,
            )

            action = policy_output.action[:, curr_index]
            obs, reward, done, _ = env.step(action)

            # add transition to the trajectory
            actions[0, curr_index] = action
            rewards[0, curr_index] = reward

            if curr_index < max_traj_len - 1:
                observations[0, curr_index + 1] = obs
                mask[0, curr_index + 1] = 1.0

            transition = Transition(
                observation=obs,
                action=action,
                reward=reward,
                done=done,
            )

            transitions.append(transition)
            curr_index += 1

            if not done[0]:
                ep_returns[episode] += reward[0]
                ep_lengths[episode] += 1

    # import ipdb

    # ipdb.set_trace()

    rollout_time = time.time() - rollout_time

    eval_metrics = {
        "episode_return": jnp.mean(ep_returns),
        "avg_length": jnp.mean(ep_lengths),
        "rollout_time": rollout_time / config.num_eval_rollouts,
    }
    return eval_metrics, transitions


def run_rollouts_procgen_helper(
    rng, agent, config: ConfigDict, env, env_id: str, **kwargs
):
    # gym version needs to be gym==0.23.1 for this to work
    logging.info("rollout procgen...")
    env = make_procgen_envs(training=False, **config.env)
    obs = env.reset()

    rng, policy_rng = jax.random.split(rng, 2)

    dones = jnp.zeros((config.num_eval_rollouts,))
    ep_returns = jnp.zeros((config.num_eval_rollouts,))
    ep_lengths = jnp.zeros((config.num_eval_rollouts,))
    transitions = []

    rollout_time = time.time()

    while not jnp.all(dones):
        # reached max timesteps, break
        done = jnp.logical_or(dones, ep_lengths >= config.env.steps_per_rollout)
        if done.all():
            break

        obs = normalize_obs(obs.astype(jnp.float32))
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
                if ep_lengths[i] < config.env.steps_per_rollout:
                    ep_returns = ep_returns.at[i].set(ep_returns[i] + reward[i])
                    ep_lengths = ep_lengths.at[i].set(ep_lengths[i] + 1)
            else:
                dones = dones.at[i].set(1)

    rollout_time = time.time() - rollout_time

    eval_metrics = {
        "episode_return": jnp.mean(ep_returns),
        "avg_length": jnp.mean(ep_lengths),
        "rollout_time": rollout_time / config.num_eval_rollouts,
    }
    return eval_metrics, transitions
