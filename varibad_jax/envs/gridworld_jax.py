import jax
import numpy as np
import jax.numpy as jnp
import itertools
from typing import List
import gymnasium as gym
from flax import struct
from gymnasium.utils import seeding
from xminigrid.types import TimeStep, StepType, EnvCarryT
from xminigrid.environment import EnvParamsT, EnvParams


class GridNaviEnvParams(struct.PyTreeNode):
    possible_goals: jnp.ndarray
    grid_size: int = 5
    max_episode_steps: int = 15
    num_actions: int = 5
    task_dim: int = 2


class State(struct.PyTreeNode):
    key: jax.Array
    step_num: jax.Array
    goal: jax.Array


class GridNavi:
    """Grid navigation environment in BRAX."""

    def __init__(self, **kwargs):
        super().__init__()

        # always starting at the bottom right
        self.init_state = jnp.array([0, 0])

    def action_space(self, params: EnvParamsT):
        # noop, up, right, down, left
        action_space = gym.spaces.Discrete(params.num_actions)
        return action_space

    def observation_space(self, params: EnvParamsT):
        return gym.spaces.Box(low=0, high=params.grid_size - 1, shape=(2,))

    def num_actions(self, params: EnvParamsT):
        return params.num_actions

    def observation_shape(self, params: EnvParamsT):
        return (2,)

    def possible_goals(self, grid_size: int):
        possible_goals = jnp.dstack(
            jnp.meshgrid(jnp.arange(grid_size), jnp.arange(grid_size))
        ).reshape(-1, 2)
        # remove (0,0), (0,1), (1,0), and (1,1)
        possible_goals = jnp.delete(possible_goals, jnp.array([0, 1, 5, 6]), axis=0)
        possible_goals = jnp.array(possible_goals)
        return possible_goals

    def default_params(self, **kwargs) -> GridNaviEnvParams:
        default_params = GridNaviEnvParams(possible_goals=None, grid_size=5)
        return default_params.replace(**kwargs)

    def reset(self, params: EnvParamsT, rng: jax.Array):
        init_xy = jnp.array(self.init_state)
        obs = init_xy

        # sample a random goal
        possible_goals = params.possible_goals
        indx = jax.random.randint(rng, (1,), 0, len(possible_goals))[0]
        goal = possible_goals[indx]

        # sample a new task from num_cells
        state = State(key=rng, step_num=jnp.array(0), goal=goal)
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=obs,
        )
        return timestep

    def step(self, params: EnvParamsT, timestep: TimeStep, action: jax.Array):
        curr_obs = timestep.observation

        # Define the possible actions: up, down, left, right
        # 0 = right, 1 = left, 2 = down, 3 = up, 4 = noop
        actions = jnp.array([(0, 1), (0, -1), (-1, 0), (1, 0), (0, 0)])

        new_obs = curr_obs + actions[action]
        # clip the new state to be within the grid boundaries
        new_obs = jnp.clip(
            new_obs,
            jnp.array([0, 0]),
            jnp.array([params.grid_size, params.grid_size]) - 1,
        )
        reached_goal = jnp.array_equal(timestep.state.goal, new_obs)
        reward = jnp.where(reached_goal, 1.0, -0.1)

        new_state = timestep.state.replace(
            step_num=timestep.state.step_num + 1,
        )
        terminated = False
        truncated = jnp.equal(new_state.step_num, params.max_episode_steps)
        step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_obs,
        )

        return timestep
