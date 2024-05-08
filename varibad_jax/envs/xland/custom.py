import jax
import jax.numpy as jnp

from xminigrid.envs.xland import XLandMiniGrid
from xminigrid.types import EnvCarryT, IntOrArray, State, StepType, TimeStep
from xminigrid.core.observation import transparent_field_of_view
from xminigrid.environment import Environment, EnvParamsT


class CustomXLandMiniGrid(XLandMiniGrid):
    def reset(self, params: EnvParamsT, key: jax.Array, state=None):
        if state is None:
            state = self._generate_problem(params, key)

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
