import functools
import os
from typing import Dict

from acme import wrappers
import dm_env
from dm_env import specs
from dopamine.discrete_domains import atari_lib


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


def environment(game: str) -> dm_env.Environment:
    """Atari environment."""
    env = atari_lib.create_atari_environment(game_name=game, sticky_actions=True)
    env = AtariDopamineWrapper(env)
    env = wrappers.FrameStackingWrapper(env, num_frames=4)
    return wrappers.SinglePrecisionWrapper(env)


if __name__ == "__main__":
    import numpy as np

    env = environment("Pong")
    obs = env.reset()
    timestep = env.step(np.array([0]))
