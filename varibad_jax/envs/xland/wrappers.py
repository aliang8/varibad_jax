import jax
import jax.numpy as jnp
from xminigrid.wrappers import Wrapper


class FullyObservableWrapper(Wrapper):
    def full_obs(self, timestep):
        # modify observation to be the full grid
        grid = timestep.state.grid
        grid_shape = grid.shape

        # add extra dimension for the agent's dir
        full_obs = jnp.zeros((grid_shape[0], grid_shape[1], 3))
        full_obs = full_obs.at[:, :, :-1].set(grid)

        agent_pos = timestep.state.agent.position
        agent_dir = timestep.state.agent.direction

        # agent tile + red color
        full_obs = full_obs.at[agent_pos[0], agent_pos[1]].set(
            jnp.array([13, 1, agent_dir])
        )
        return full_obs

    def reset(self, params, key):
        timestep = self._env.reset(params, key)
        full_obs = self.full_obs(timestep)
        timestep = timestep.replace(observation=full_obs)
        return timestep

    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        full_obs = self.full_obs(timestep)
        timestep = timestep.replace(observation=full_obs)
        return timestep

    def observation_shape(self, env_params):
        return (env_params.height, env_params.width, 3)
