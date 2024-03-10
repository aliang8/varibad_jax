from collections import deque
import copy
from enum import Enum
import itertools
import math
import random
from typing import List, Tuple
from absl import logging
import gymnasium as gym
from gym import spaces
from ml_collections.config_dict import ConfigDict
import numpy as np

class Action(Enum):
  NOOP: int = 0
  UP: int = 1
  RIGHT: int = 2
  DOWN: int = 3
  LEFT: int = 4

class GridNavi(gym.Env):
  """Grid navigation environment."""

  def __init__(self, env_config: ConfigDict, **kwargs):
    super().__init__()

    self.seed()
    self.env_config = env_config
    self._max_episode_steps = self.env_config.steps_per_rollout
    self.num_cells = self.env_config.num_cells
    self.task_dim = 1
    self.step_count = 0

    obs_shape = (1,)
    self.observation_space = spaces.Box(
        low=0, high=self.num_cells - 1, shape=obs_shape
    )

    # noop, up, right, down, left
    self.action_space = spaces.Discrete(5)

    # possible starting states
    # always starting at the bottom right
    self.starting_state = (0, 0)

    # goals can be anywhere except on possible starting states and immediately
    # around it
    self.possible_goals = list(
        itertools.product(range(self.num_cells), repeat=2)
    )
    self.possible_goals.remove((0, 0))
    self.possible_goals.remove((0, 1))
    self.possible_goals.remove((1, 1))
    self.possible_goals.remove((1, 0))

    # reset the environment state
    self._env_state = np.array(self.starting_state)
    self._goal = None

  def reset_task(self, task=None, **kwargs):
    if task is None:
      self._goal = np.array(random.choice(self.possible_goals))
    else:
      self._goal = np.array(task)
    return self._goal

  def get_obs(self):
    return self.coord_to_id(self._env_state)

  def get_task(self):
    task = self.coord_to_id(self._goal)
    return np.array(task)

  def get_all_goals(self):
    return self.get_task()

  def reset(self, **kwargs):
    self.step_count = 0
    self._env_state = np.array(self.starting_state)

    # get a new goal
    self.reset_task(session_count=0)
    return self.get_obs()

  def state_transition(self, action):
    if action == 1:  # up
      self._env_state[1] = min([self._env_state[1] + 1, self.num_cells - 1])
    elif action == 2:  # right
      self._env_state[0] = min([self._env_state[0] + 1, self.num_cells - 1])
    elif action == 3:  # down
      self._env_state[1] = max([self._env_state[1] - 1, 0])
    elif action == 4:  # left
      self._env_state[0] = max([self._env_state[0] - 1, 0])

  def reached_goal(self, state):
    return np.array_equal(state, self.get_task())

  def step(self, action):
    if isinstance(action, np.ndarray) and action.ndim == 1:
      action = action[0]
    assert self.action_space.contains(action)

    done = False

    # perform state transition
    self.state_transition(action)
    next_state = self.get_obs()

    # check if maximum step limit is reached
    self.step_count += 1
    if self.step_count >= self._max_episode_steps:
      done = True

    # compute reward
    reached_goal = self.reached_goal(next_state)

    if reached_goal:
      reward = 1.0
    else:
      # some small time penalty
      reward = -0.1

    info = {'task': self.get_task()}
    return next_state, reward, done, info

  def coord_to_id(self, goals):
    mat = np.arange(0, self.num_cells**2, dtype=np.float32).reshape(
        (self.num_cells, self.num_cells)
    )
    if isinstance(goals, list) or isinstance(goals, tuple):
      goals = np.array(goals)

    if goals.ndim == 1:
      goals = goals[np.newaxis]

    goal_shape = goals.shape
    if len(goal_shape) > 2:
      goals = goals.reshape(-1, goals.shape[-1])

    classes = mat[goals[:, 0], goals[:, 1]]
    classes = classes.reshape(goal_shape[:-1])

    return classes

  def id_to_task(self, classes):
    mat = np.arange(0, self.num_cells**2).reshape(
        (self.num_cells, self.num_cells)
    )
    goals = np.zeros((len(classes), 2))

    for i in range(len(classes)):
      pos = np.where(classes[i] == mat)
      goals[i, 0] = float(pos[0][0])
      goals[i, 1] = float(pos[1][0])

    return goals

  def session_reset(self, session_count: int = None, **kwargs):
    # do nothing here
    pass
