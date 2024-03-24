import gymnasium as gym
import numpy as np
import jax
from typing import ClassVar, Optional

from brax.envs import Env, State, Wrapper
from varibad_jax.envs.gridworld_jax import GridNavi
from varibad_jax.envs.wrappers import BAMDPWrapper


def make_envs(
    env_id,
    seed: int = 0,
    num_envs: int = 1,
    num_episodes_per_rollout: int = 4,
    env_kwargs=dict(),
):
    # env = gym.make(env_id)
    env = GridNavi(seed=seed, **env_kwargs)
    env = BAMDPWrapper(env, num_episodes_per_rollout=num_episodes_per_rollout)
    if num_envs > 1:
        env = VmapWrapper(env, num_envs=num_envs)
    env = GymWrapper(env, seed)
    return env


class VmapWrapper(Wrapper):
    """Vectorizes JAX env."""

    def __init__(self, env: Env, num_envs: Optional[int] = None):
        super().__init__(env)
        self.num_envs = num_envs

    def reset(self, rng: jax.Array):
        if self.num_envs is not None:
            rng = jax.random.split(rng, self.num_envs)
        return jax.vmap(self.env.reset)(rng)

    def step(self, state: State, action: jax.Array):
        return jax.vmap(self.env.step)(state, action)


class GymWrapper(gym.Env):
    """A wrapper that converts JAX Env to one that follows Gym API."""

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env, seed: int = 0):
        self._env = env
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
        }
        self.seed(seed)
        self._state = None
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.task_dim = self._env.task_dim

        def reset(key):
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset)

        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step)

    def reset(self):
        self._state, obs, self._key = self._reset(self._key)
        # We return device arrays for pytorch users.
        return obs

    def step(self, action):
        self._state, obs, reward, done, info = self._step(self._state, action)
        # We return device arrays for pytorch users.
        return obs, reward, done, info

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode="human"):
        self._env.render(mode)

    @property
    def max_episode_steps(self):
        return self._env.max_episode_steps

    def close(self):
        self._env.close()
