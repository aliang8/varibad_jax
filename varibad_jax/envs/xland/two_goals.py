from __future__ import annotations

import jax
import jax.numpy as jnp

from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles, NUM_COLORS
from xminigrid.core.goals import AgentNearGoal
from xminigrid.core.grid import (
    coordinates_mask,
    room,
    sample_coordinates,
    sample_direction,
    vertical_line,
)
from xminigrid.core.rules import EmptyRule
from xminigrid.environment import Environment, EnvParams
from xminigrid.types import AgentState, EnvCarry, State

from varibad_jax.envs.xland.custom import CustomXLandMiniGrid


class TwoGoals(CustomXLandMiniGrid):
    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State[EnvCarry]:
        key, _key = jax.random.split(key)
        keys = jax.random.split(_key, num=5)

        # objects
        obj1 = TILES_REGISTRY[Tiles.BALL, Colors.RED]
        obj2 = TILES_REGISTRY[Tiles.BALL, Colors.BLUE]

        # pick one as goal
        goal_indx = jax.random.randint(keys[2], shape=(), minval=0, maxval=2)
        goal_tile = jax.lax.switch(
            goal_indx,
            (lambda: obj1, lambda: obj2),
        )

        grid = room(params.height, params.width)
        obj1_coords, obj2_coords, agent_coords = sample_coordinates(
            keys[1], grid, num=3
        )

        grid = grid.at[obj1_coords[0], obj1_coords[1]].set(obj1)
        grid = grid.at[obj2_coords[0], obj2_coords[1]].set(obj2)

        _goal_encoding = AgentNearGoal(tile=goal_tile).encode()
        _rule_encoding = EmptyRule().encode()[None, ...]

        agent = AgentState(position=agent_coords, direction=sample_direction(keys[2]))
        state = State(
            key=key,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            goal_encoding=_goal_encoding,
            rule_encoding=_rule_encoding,
            carry=EnvCarry(),
        )
        return state
