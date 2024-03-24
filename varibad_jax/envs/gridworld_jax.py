import jax
import numpy as np
import jax.numpy as jnp
import itertools
from typing import List
from brax.envs import State
import gymnasium as gym


class GridNavi:
    """Grid navigation environment in BRAX."""

    def __init__(self, seed: int = 0, num_cells: int = 5, episode_length: int = 15):
        super().__init__()
        self._max_episode_steps = episode_length
        self.num_cells = num_cells

        obs_shape = (2,)  # x, y coordinates
        self.observation_space = gym.spaces.Box(
            low=0, high=self.num_cells - 1, shape=obs_shape
        )

        # noop, up, right, down, left
        self.action_space = gym.spaces.Discrete(5)

        # possible starting states
        # always starting at the bottom right
        self.init_state = jnp.array([0, 0])

        # goals can be anywhere except on possible starting states and immediately
        # around it
        self.possible_goals = list(itertools.product(range(self.num_cells), repeat=2))
        self.possible_goals.remove((0, 0))
        self.possible_goals.remove((0, 1))
        self.possible_goals.remove((1, 1))
        self.possible_goals.remove((1, 0))
        self.possible_goals = jnp.array(self.possible_goals)
        self.np_random = np.random.default_rng(seed=seed)

    def reset(self, rng: jax.Array):
        init_xy = jnp.array(self.init_state)
        obs = init_xy
        reward, done, zero = jnp.zeros(3)

        # sample a random goal
        indx = self.np_random.integers(0, len(self.possible_goals))
        goal = self.possible_goals[indx]

        # sample a new task from num_cells
        info = {
            "goal": goal,
            "timestep": 0,
        }
        metrics = {"reward": reward}

        return State(
            pipeline_state=None,
            obs=obs,
            reward=reward.astype(jnp.float32),
            done=done,
            metrics=metrics,
            info=info,
        )

    def step(self, state: State, action: jax.Array):
        curr_obs = state.obs

        # Define the possible actions: up, down, left, right
        actions = jnp.array([(0, 1), (0, -1), (-1, 0), (1, 0), (0, 0)])

        new_xy = curr_obs + actions[action[0]]
        # clip the new state to be within the grid boundaries
        new_xy = jnp.clip(
            new_xy, jnp.array([0, 0]), jnp.array([self.num_cells, self.num_cells]) - 1
        )
        obs = new_xy
        reached_goal = jnp.array_equal(state.info["goal"], obs)
        reward = jnp.where(reached_goal, 1.0, -0.1)

        metrics = {"reward": reward, "reached_goal": reached_goal}
        state.info["timestep"] = state.info["timestep"] + 1
        done = state.info["timestep"] >= self._max_episode_steps
        return state.replace(obs=obs, reward=reward, done=done, metrics=metrics)

    # def xy_to_id(self, xy: jax.Array):
    #     mat = jnp.arange(0, self.num_cells**2).reshape((self.num_cells, self.num_cells))
    #     id = mat[xy[0], xy[1]]
    #     return id

    @property
    def max_episode_steps(self):
        return self._max_episode_steps
