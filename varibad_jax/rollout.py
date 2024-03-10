from collections import defaultdict as dd
from functools import partial
import time
from typing import Optional

from absl import logging
from flax.training.train_state import TrainState
import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
import numpy as np

from google3.experimental.posterior_transformer.agents.ppo.ppo_hk import policy_jit, select_action
from google3.experimental.posterior_transformer.envs.util import env_step, make_env, make_vec_envs, reset_env
from google3.experimental.posterior_transformer.models.helpers import decode, encode_trajectory, select_action_bc


def eval_rollout(
    config: ConfigDict,
    ts_policy: TrainState,
    ts_vae: TrainState,
    rng_seq: hk.PRNGSequence,
    goal_transition_matrix: Optional[np.ndarray] = None,
    num_rollouts: Optional[int] = None,
    num_episodes_per_rollout: Optional[int] = None,
    policy: Optional = None,
):
  num_rollouts = num_rollouts or config.exp.num_eval_rollouts
  num_episodes_per_rollout = (
      num_episodes_per_rollout or config.env.num_episodes_per_rollout
  )

  # unfreeze if already frozen
  config = ConfigDict(config)

  config.env.num_processes = num_rollouts
  config.env.num_episodes_per_rollout = num_episodes_per_rollout

  config = FrozenConfigDict(config)
  if config.vae.prior_type == "gaussian":
    latent_dim = 2 * config.vae.latent_dim
  else:
    latent_dim = config.vae.latent_dim

  # Initialize environments
  envs = make_vec_envs(
      seed=config.exp.seed,
      env_config=config.env,
      goal_transition_matrix=goal_transition_matrix,
  )

  # make sure this is frozen
  config = FrozenConfigDict(config)

  # keep track of returns for each episode in the BAMDP
  returns_per_episode = np.zeros((num_rollouts, num_episodes_per_rollout + 1))
  # this counts how often an agent has done the same task already
  task_count = np.zeros(num_rollouts, dtype=int)

  rollouts = dd(list)

  # reset
  (state, belief, task) = reset_env(envs)

  # initialize prior over latent variable
  # use all zeros as priors
  latent = np.zeros((num_rollouts, latent_dim))
  hidden_state = np.zeros((num_rollouts, config.vae.lstm_hidden_size))
  prev_goal = np.zeros((num_rollouts, envs.task_dim))
  prev_goal_embeds = np.zeros((num_rollouts, config.vae.embedding_dim))
  prev_goal_hidden_state = np.zeros((num_rollouts, config.vae.lstm_hidden_size))

  # belief is concat of mean and variance vector
  if config.vae.input_gt_belief:
    long_term_belief = np.zeros((
        num_rollouts,
        25 * config.vae.num_prev_m_condition,
    ))
  else:
    long_term_belief = np.zeros((
        num_rollouts,
        latent_dim * config.vae.num_prev_m_condition,
    ))

  done_mdp = False

  # initial state
  if len(state.shape) == 1:
    state = state[np.newaxis]

  if config.policy.pass_belief_to_policy and len(belief.shape) == 1:
    belief = belief[np.newaxis]

  encode_fn = partial(jax.jit, static_argnames="config")(
      encode_trajectory.apply
  )

  if not config.policy.rule_based:
    if config.policy.algo == "ppo":
      get_action_fn = partial(jax.jit, static_argnames="config")(
          partial(
              select_action, config=config.policy, sample=False
          )  # deterministic
      )
    elif config.policy.algo in ["bc", "iql", "cql"]:
      get_action_fn = jax.jit(select_action_bc)
  else:
    goal_preds = [np.random.randint(25) for _ in range(num_rollouts)]

  goal_states = envs.get_task()
  if len(goal_states.shape) == 1:
    goal_states = goal_states[np.newaxis]

  total_steps_per_rollout = num_episodes_per_rollout * envs._max_episode_steps
  for step in range(total_steps_per_rollout):
    # select action
    if config.policy.rule_based:
      action = policy.select_action(state, goal_preds)
    else:
      if config.policy.algo == "ppo":
        action, _, _ = get_action_fn(
            train_state=ts_policy,
            state=state,
            task=task,
            latent=latent,
            rng=next(rng_seq),
        )
      elif config.policy.algo in ["bc", "iql", "cql"]:
        action = get_action_fn(
            train_state=ts_policy, obs=state, latent=latent, key=next(rng_seq)
        )

    # can we compute the value for every state here
    # all_states = jnp.arange(25)[:, np.newaxis]
    # all_states = jnp.concatenate([all_states, state[:, 1:].repeat(25, 0)], axis=-1)
    # all_latents = latent.repeat(25, 0)

    # _, _, all_values = get_action_fn(
    #     train_state=ts_policy,
    #     state=all_states,
    #     task=task,
    #     latent=all_latents,
    #     rng=next(rng_seq),
    # )

    # This needs to be num_envs x 1
    if config.policy.action_type == "discrete":
      action = np.array(action, dtype=int)[:, np.newaxis]
    # print(step, " state: ", state, "action: ", action)

    curr_goal = envs.get_task()
    all_goals = envs.get_all_goals()

    (next_state, belief, task), [rew_raw, rew_norm], done, infos = env_step(
        envs, action
    )

    # extract step metadata
    keys = ["done_mdp", "session_mask", "opt_acts", "opt_rews"]
    infos_dict = {} 
    for k in keys:
      if k in infos[0].keys():
        infos_dict[k] = np.array([info[k] for info in infos])

        # add extra dim
        infos_dict[k] = infos_dict[k][..., np.newaxis]

    # import ipdb; ipdb.set_trace()
    done_mdp = infos_dict["done_mdp"].astype(bool).squeeze()
    # update the previous goal state when there is a new session
    next_goal = envs.get_task()
    prev_goal[done_mdp] = next_goal[done_mdp]

    # increase rewards
    returns_per_episode[range(num_rollouts), task_count] += rew_raw

    # add extra dimension
    if len(next_state.shape) == 1:
      next_state = next_state[np.newaxis]

    reward = np.array(rew_raw, dtype=float)[:, np.newaxis]
    done = np.array(done, dtype=bool)[:, np.newaxis]
    belief = np.array(belief, dtype=float)

    # update hidden state
    encode_outputs = encode_fn(
        ts_vae.params,
        next(rng_seq),
        config=config.vae,
        states=next_state,
        actions=action,
        rewards=reward,
        hidden_state=hidden_state,
        belief=belief,
        done_mdp=done_mdp[:, np.newaxis],
        prev_goals=prev_goal,
        prev_goal_embeds=prev_goal_embeds,
        prev_goal_hs=prev_goal_hidden_state
    )

    latent = encode_outputs.latent
    hidden_state = encode_outputs.hidden_state
    if encode_outputs.prev_goal_hidden_state is not None:
      prev_goal_embeds = jax.lax.stop_gradient(
          encode_outputs.prev_goal_embeds
      )
      prev_goal_hidden_state = jax.lax.stop_gradient(
          encode_outputs.prev_goal_hidden_state
      )

    if config.exp.save_video and "grid" not in config.env.env_name.lower():
      # should be a list of images
      frames = envs.render("rgb_array")

      # make the image smaller
      import cv2

      w, h = frames[0].shape[:2]

      for idx, frame in enumerate(frames):
        frames[idx] = cv2.resize(
            frame, (int(h // 3), int(w // 3)), interpolation=cv2.INTER_AREA
        )

      frames = np.array(frames)
      rollouts["frames"].append(frames)

    rollouts["goals"].append(curr_goal)
    rollouts["states"].append(state)
    rollouts["beliefs"].append(belief)
    rollouts["next_states"].append(next_state)
    rollouts["tasks"].append(task)
    rollouts["actions"].append(action)
    rollouts["dones"].append(done)
    rollouts["rewards"].append(reward)
    rollouts["rewards_norm"].append(rew_norm)
    rollouts["infos"].append(infos)
    rollouts["returns"].append(returns_per_episode.copy())
    rollouts["hidden_state"].append(hidden_state)
    rollouts["latent"].append(latent)
    # rollouts["all_values"].append(all_values)
    rollouts["possible_goals"].append(all_goals)
    for k in infos_dict:
      rollouts[k].append(infos_dict[k])

    # for the ant environment, let's collect the pos information for plotting purposes
    if "ant" in config.env.env_name.lower():
      ant_pos = np.array([info["ant_pos"] for info in infos])
      rollouts["ant_pos"].append(ant_pos)

    if (
        hasattr(encode_outputs, "termination_pred")
        and encode_outputs.termination_pred is not None
    ):
      rollouts["termination_pred"].append(encode_outputs.termination_pred)

    # increase the counter for processes that finish the mdp
    for i in np.argwhere(np.array(done_mdp)).flatten():
      task_count[i] = min(task_count[i] + 1, num_episodes_per_rollout)

    # reset environments where the task is complete
    if done.sum() > 0:
      done_indices = np.argwhere(done.flatten()).astype(int).flatten()

      (next_state, belief, task) = reset_env(envs, done_indices, state)

    state = next_state

    if len(state.shape) == 1:
      state = state[np.newaxis]
    if config.policy.pass_belief_to_policy and len(belief.shape) == 1:
      belief = belief[np.newaxis]

    # # update the long_term_belief after each episode ends
    # if config.vae.model_cls == "varigood":
    #   if config.env.varying_episode_length:
    #     # termination_pred = encode_outputs.termination_pred
    #     long_term_belief = encode_outputs.long_term_belief
    #     # mask = termination_pred > 0.5
    #   else:
    #     if ((step + 1) % env._max_episode_steps) == 0:
    #       long_term_belief = latent

  for k, v in rollouts.items():
    if k != "infos":
      rollouts[k] = np.array(v)

  # close the env
  envs.close()
  returns_per_episode = returns_per_episode[:, :num_episodes_per_rollout]
  return returns_per_episode, rollouts


def eval_rollout_transformer(
    env,
    config: FrozenConfigDict,
    ts_policy: TrainState,
    ts_vae: TrainState,
    rng_seq: hk.PRNGSequence,
    num_rollouts: Optional[int] = None,
    num_episodes_per_rollout: Optional[int] = None,
):
  """Different from evaluate for LSTM because we have to concatenate

  the full history of trajectory for Transformer. In LSTM, the hidden
  state will keep track of history information.
  """
  # make sure this is frozen
  config = FrozenConfigDict(config)

  num_rollouts = num_rollouts or config.exp.num_eval_rollouts
  num_episodes_per_rollout = (
      num_episodes_per_rollout or config.env.num_episodes_per_rollout
  )

  # keep track of returns for each episode in the BAMDP
  returns_per_episode = np.zeros((num_rollouts, num_episodes_per_rollout + 1))
  # this counts how often an agent has done the same task already
  task_count = np.zeros(num_rollouts, dtype=int)

  rollouts = dd(list)

  # reset
  (state, belief, task) = reset_env(env)

  # initialize prior over latent variable
  # use all zeros as priors
  if config.vae.prior_type == "gaussian":
    latent_dim = 2 * config.vae.latent_dim
  else:
    latent_dim = config.vae.latent_dim

  latent = jnp.zeros((num_rollouts, latent_dim))
  hidden_state = jnp.zeros((num_rollouts, config.vae.embedding_dim))

  # belief is concat of mean and variance vector
  long_term_belief = jnp.zeros((num_rollouts, latent_dim))

  done_mdp = False

  # Transformer encoder is also batch first
  # But we want to make it fixed size for JAX jit
  total_ts = num_episodes_per_rollout * env._max_episode_steps
  states = jnp.zeros((num_rollouts, total_ts, state.shape[-1]))
  beliefs = jnp.zeros((num_rollouts, total_ts, belief.shape[-1]))
  actions = jnp.zeros((num_rollouts, total_ts, 1))
  rewards = jnp.zeros((num_rollouts, total_ts, 1))
  mask = jnp.zeros((num_rollouts, total_ts, 1))

  states = states.at[:, 0].set(state)
  beliefs = beliefs.at[:, 0].set(belief)
  mask = mask.at[:, 0].set(1)

  if len(state.shape) == 1:
    state = state[jnp.newaxis]
  if len(belief.shape) == 1:
    belief = belief[jnp.newaxis]

  encode_fn = partial(jax.jit, static_argnames="config")(
      encode_trajectory.apply
  )
  for step in range(total_ts):
    # select action
    action, _, _ = select_action(
        train_state=ts_policy,
        config=config.policy,
        state=state,
        belief=beliefs,
        task=task,
        latent=latent,
        rng=next(rng_seq),
        sample=False,  # make deterministic
    )

    # action = action[-1] # take the final action, assuming this is [T, B, action_dim]

    # This needs to be num_envs x 1
    # Also needs to be an np array and not a jax array
    action = np.array(action, dtype=int)[:, np.newaxis]

    # observe reward and next obs
    goal = env.get_task()
    if len(goal.shape) == 1:
      goal = goal[jnp.newaxis]
    rollouts["goals"].append(goal)
    (next_state, belief, task), reward, done, infos = env_step(env, action)
    done_mdp = [info["done_mdp"] for info in infos]

    # increase rewards
    returns_per_episode[range(num_rollouts), task_count] += reward

    # add extra dimension
    if len(next_state.shape) == 1:
      next_state = next_state[jnp.newaxis]

    reward = jnp.array(reward, dtype=float)[:, jnp.newaxis]

    # We want to concatenate the trajectory information
    actions = actions.at[:, step].set(action)
    rewards = rewards.at[:, step].set(reward)

    # update hidden state
    encode_outputs = encode_fn(
        ts_vae.params,
        next(rng_seq),
        config=config.vae,
        states=states,
        actions=actions,
        rewards=rewards,
        hidden_state=None,  # no hidden state for varibad transformer TODO
        mask=mask,
        long_term_belief=long_term_belief
        if config.vae.model_cls == "varigood"
        else None,
    )

    latent = encode_outputs.latent[:, -1]
    rollouts["states"].append(state)
    rollouts["beliefs"].append(belief)
    rollouts["next_states"].append(next_state)
    rollouts["tasks"].append(task)
    rollouts["actions"].append(action)
    rollouts["dones"].append(done)
    rollouts["rewards"].append(reward)
    rollouts["infos"].append(infos)
    rollouts["returns"].append(returns_per_episode.copy())
    rollouts["hidden_state"].append(hidden_state)
    rollouts["latent"].append(latent)
    rollouts["done_mdp"].append(done_mdp)
    rollouts["possible_goals"].append(all_goals)

    if (
        hasattr(encode_outputs, "termination_pred")
        and encode_outputs.termination_pred is not None
    ):
      rollouts["termination_pred"].append(encode_outputs.termination_pred)

    if (
        hasattr(encode_outputs, "long_term_belief")
        and encode_outputs.long_term_belief is not None
    ):
      rollouts["long_term_belief"].append(long_term_belief)

    # increase the counter for processes that finish the mdp
    for i in np.argwhere(np.array(done_mdp)).flatten():
      task_count[i] = min(task_count[i] + 1, num_episodes_per_rollout)

    # reset environments where the task is complete
    if done.sum() > 0:
      done_indices = np.argwhere(done.flatten()).astype(int).flatten()
      (next_state, belief, task) = reset_env(env, done_indices, state)

    # Don't do this
    state = next_state

    if len(state.shape) == 1:
      state = state[jnp.newaxis]
    if len(belief.shape) == 1:
      belief = belief[jnp.newaxis]

    beliefs = beliefs.at[:, step + 1].set(belief)
    states = states.at[:, step + 1].set(next_state)
    mask = mask.at[:, step + 1].set(1)

  # clean up
  env.close()

  for k, v in rollouts.items():
    if k != "infos":
      rollouts[k] = np.array(v)

  returns_per_episode = returns_per_episode[:, :num_episodes_per_rollout]
  return returns_per_episode, rollouts
