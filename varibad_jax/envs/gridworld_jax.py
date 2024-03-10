import jax
import numpy as np
import jax.numpy as jnp
import itertools
from typing import List
from brax.envs import State
import gymnasium as gym


class GridNavi:
    """Grid navigation environment in BRAX."""

    def __init__(self, episode_length: int = 15, num_cells: int = 5):
        super().__init__()
        self._max_episode_steps = episode_length
        self.num_cells = num_cells

        obs_shape = (1,)
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

    def _get_obs(self, xy: jnp.ndarray):
        return self.xy_to_id(xy)[..., jnp.newaxis]

    def reset(self, rng: jax.Array):
        init_xy = jnp.array(self.init_state)
        obs = self._get_obs(init_xy)
        reward, done, zero = jnp.zeros(3)

        # sample a random goal
        indx = np.random.randint(0, len(self.possible_goals))
        goal = self.possible_goals[indx]
        task = self._get_obs(goal)

        # sample a new task from num_cells
        info = {
            "xy_coord": jnp.zeros(2).astype(jnp.int32),
            "goal": goal,
            "task": task,
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

    def reached_goal(self, state):
        return np.array_equal(state, self.get_task())

    def step(self, state: State, action: jax.Array):
        curr_obs = state.info["xy_coord"]

        # Define the possible actions: up, down, left, right
        actions = jnp.array([(0, 1), (0, -1), (-1, 0), (1, 0), (0, 0)])

        new_xy = curr_obs + actions[action[0]]
        # clip the new state to be within the grid boundaries
        new_xy = jnp.clip(
            new_xy, jnp.array([0, 0]), jnp.array([self.num_cells, self.num_cells]) - 1
        )
        obs = self._get_obs(new_xy)
        reached_goal = jnp.array_equal(state.info["task"], obs)
        reward = jnp.where(reached_goal, 1.0, -0.1)

        metrics = {"reward": reward, "reached_goal": reached_goal}
        state.info["xy_coord"] = new_xy
        state.info["timestep"] = state.info["timestep"] + 1
        done = state.info["timestep"] >= self._max_episode_steps
        return state.replace(obs=obs, reward=reward, done=done, metrics=metrics)

    def xy_to_id(self, xy: jax.Array):
        mat = jnp.arange(0, self.num_cells**2).reshape((self.num_cells, self.num_cells))
        id = mat[xy[0], xy[1]]
        return id

    @property
    def max_episode_steps(self):
        return self._max_episode_steps
