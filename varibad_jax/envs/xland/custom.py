import jax
import jax.numpy as jnp
from functools import partial

from xminigrid.envs.xland import XLandMiniGrid
from xminigrid.types import EnvCarryT, IntOrArray, State, StepType, TimeStep, AgentState
from xminigrid.core.observation import transparent_field_of_view
from xminigrid.environment import Environment, EnvParamsT
from xminigrid.core.grid import sample_coordinates, sample_direction


class CustomXLandMiniGrid(XLandMiniGrid):
    @partial(jax.jit, static_argnames=("self", "randomize_agent"))
    def reset(
        self,
        params: EnvParamsT,
        key: jax.Array,
        state=None,
        randomize_agent: bool = False,
    ):
        if state is None:
            state = self._generate_problem(params, key)

        if randomize_agent:
            # still randomize agent initial location and direction
            key, coords_key, dir_key = jax.random.split(key, num=3)
            grid = state.grid
            positions = sample_coordinates(coords_key, grid, num=1)
            agent = AgentState(
                position=positions[-1], direction=sample_direction(dir_key)
            )
        else:
            agent = state.agent

        state = state.replace(agent=agent)
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=transparent_field_of_view(
                state.grid, state.agent, params.view_size, params.view_size
            ),
        )
        return timestep
