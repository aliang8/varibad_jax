import functools
import os
from typing import Dict, Optional, Tuple, Any
import tensorflow as tf

import gym
from acme import wrappers
import dm_env
from dm_env import specs
from dopamine.discrete_domains import atari_lib
from varibad_jax.envs.atari.atari_constants import *
import jax
import jax.numpy as jnp


class AtariDopamineWrapper(dm_env.Environment):
    """Wrapper for Atari Dopamine environmnet."""

    def __init__(self, env, max_episode_steps=108000):
        self._env = env
        self._max_episode_steps = max_episode_steps
        self._episode_steps = 0
        self._reset_next_episode = True

    def reset(self):
        self._episode_steps = 0
        self._reset_next_step = False
        observation = self._env.reset()
        return dm_env.restart(observation.squeeze(-1))

    def step(self, action):
        if self._reset_next_step:
            return self.reset()

        observation, reward, terminal, _ = self._env.step(action.item())
        observation = observation.squeeze(-1)
        discount = 1 - float(terminal)
        self._episode_steps += 1
        if terminal:
            self._reset_next_episode = True
            return dm_env.termination(reward, observation)
        elif self._episode_steps == self._max_episode_steps:
            self._reset_next_episode = True
            return dm_env.truncation(reward, observation, discount)
        else:
            return dm_env.transition(reward, observation, discount)

    def observation_spec(self):
        space = self._env.observation_space
        return specs.Array(space.shape[:-1], space.dtype)

    def action_spec(self):
        return specs.DiscreteArray(self._env.action_space.n)

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space


def create_atari_env(game: str) -> dm_env.Environment:
    """Atari environment."""
    env = AtariEnvWrapper(game)
    # env = atari_lib.create_atari_environment(game_name=game, sticky_actions=False)
    # env = AtariDopamineWrapper(env)
    # env = wrappers.FrameStackingWrapper(env, num_frames=4)
    return env
    # return wrappers.SinglePrecisionWrapper(env)


def _process_observation(obs):
    """Process observation."""
    # Apply jpeg auto-encoding to better match observations in the dataset.
    return tf.io.decode_jpeg(tf.io.encode_jpeg(obs)).numpy()


class AtariEnvWrapper:
    """Environment wrapper with a unified API.

    Taken from: https://github.com/google-research/google-research/blob/master/multi_game_dt/Multi_game_decision_transformers_public_colab.ipynb
    """

    def __init__(self, game_name: str, full_action_set: Optional[bool] = True):
        # Disable randomized sticky actions to reduce variance in evaluation.
        self._env = atari_lib.create_atari_environment(game_name, sticky_actions=False)
        self.game_name = game_name
        self.full_action_set = full_action_set

    @property
    def observation_space(self) -> gym.Space:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        if self.full_action_set:
            return gym.spaces.Discrete(len(_FULL_ACTION_SET))
        return self._env.action_space

    def reset(self) -> np.ndarray:
        """Reset environment and return observation."""
        import ipdb

        ipdb.set_trace()
        return _process_observation(self._env.reset())

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Any]:
        """Step environment and return observation, reward, done, info."""
        if self.full_action_set:
            # atari_py library expects limited action set, so convert to limited.
            action = FULL_ACTION_TO_LIMITED_ACTION[self.game_name][action]
        obs, rew, done, info = self._env.step(action)
        obs = _process_observation(obs)
        return obs, rew, done, info


def _batch_rollout(rng, envs, num_steps=2500, log_interval=None):
    """Roll out a batch of environments under a given policy function."""
    # observations are dictionaries. Merge into single dictionary with batched
    # observations.
    obs_list = [env.reset() for env in envs]
    num_batch = len(envs)
    obs = jax.tree_util.tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
    ret = np.zeros([num_batch, 8])
    done = np.zeros(num_batch, dtype=np.int32)
    rew_sum = np.zeros(num_batch, dtype=np.float32)
    frames = []

    for t in range(num_steps):
        # Collect observations
        frames.append(np.concatenate([o[-1, ...] for o in obs_list], axis=1))
        done_prev = done

        # actions, rng = policy_fn(rng, obs)
        actions = jnp.zeros((len(envs),), dtype=jnp.int32)

        # Collect step results and stack as a batch.
        step_results = [env.step(act) for env, act in zip(envs, actions)]
        obs_list = [result[0] for result in step_results]
        obs = jax.tree_util.tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
        rew = np.stack([result[1] for result in step_results])
        done = np.stack([result[2] for result in step_results])
        # Advance state.
        done = np.logical_or(done, done_prev).astype(np.int32)
        rew = rew * (1 - done)
        rew_sum += rew
        if log_interval and t % log_interval == 0:
            print("step: %d done: %s reward: %s" % (t, done, rew_sum))
        # Don't continue if all environments are done.
        if np.all(done):
            break
    return rew_sum, frames, rng


if __name__ == "__main__":
    import numpy as np

    env = create_atari_env("Pong")
    obs = env.reset()
    # timestep = env.step(np.array([0]))
    # print(env.observation_spec())
    # print(env.action_spec().num_values)
    num_envs = 5
    env_batch = [create_atari_env("Pong") for i in range(num_envs)]

    import ipdb

    ipdb.set_trace()
    rng = jax.random.PRNGKey(0)
    rew_sum, frames, rng = _batch_rollout(
        rng, env_batch, num_steps=5000, log_interval=100
    )
