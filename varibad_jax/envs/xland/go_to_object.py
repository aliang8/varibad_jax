from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Optional

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
from flax import struct


from varibad_jax.envs.xland.custom import CustomXLandMiniGrid


class EnvParams(struct.PyTreeNode):
    # WARN: pytree_node=False, so you CAN NOT vmap on them!
    # You can add pytree node params, but be careful and
    # test that your code will work under jit.
    # Spoiler: probably it will not :(
    height: int = struct.field(pytree_node=False, default=9)
    width: int = struct.field(pytree_node=False, default=9)
    view_size: int = struct.field(pytree_node=False, default=7)
    max_steps: Optional[None] = struct.field(pytree_node=False, default=None)
    render_mode: str = struct.field(pytree_node=False, default="rgb_array")
    num_distractors: int = struct.field(pytree_node=False, default=1)
    goal_object: int = struct.field(pytree_node=False, default=0)


class GoToObject(CustomXLandMiniGrid):
    def default_params(self, **kwargs) -> EnvParams:
        params = EnvParams(height=9, width=9)
        params = params.replace(**kwargs)

        if params.max_steps is None:
            # formula directly taken from MiniGrid
            params = params.replace(max_steps=4 * (params.height * params.width))
        return params
    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State[EnvCarry]:
        key, _key = jax.random.split(key)
        keys = jax.random.split(_key, num=6)

        # pick four random color for the tiles without replacement
        # colors = jax.random.permutation(keys[0], jnp.arange(1, NUM_COLORS))[:4]

        valid_colors = jnp.array(
            [
                Colors.RED,
                Colors.GREEN,
                Colors.BLUE,
                Colors.YELLOW,
                Colors.ORANGE,
                Colors.PURPLE,
                Colors.WHITE,
                Colors.BROWN,
                Colors.PINK,
            ]
        )

        valid_objects = jnp.array(
            [
                Tiles.BALL, 
                Tiles.SQUARE, 
                Tiles.DOOR_LOCKED, 
                Tiles.KEY,
            ]
        )        
        valid_colors = jax.random.choice(keys[0], valid_colors, shape=(6,), replace=False)
        # sample distractors + goal colors
        colors = jax.random.choice(keys[1], valid_colors, shape=(params.num_distractors+1,), replace=False)

        goal_tile = TILES_REGISTRY[valid_objects[params.goal_object], colors[0]]  
        # delete goal object from valid_objects
        valid_objects = jnp.delete(valid_objects, params.goal_object) 

        # pick objects for distractors
        objs = jax.random.choice(keys[2], valid_objects, shape=(params.num_distractors,), replace=False)

        distractor_tiles = []
        for i in range(params.num_distractors):
            distractor_tiles.append(TILES_REGISTRY[objs[i], colors[i+1]])

        grid = room(params.height, params.width)

        coords = sample_coordinates(
            keys[3], grid, num=params.num_distractors+1
        )

        mask = jnp.ones((grid.shape[0], grid.shape[1]), dtype=jnp.bool_)

        #goal_tile = TILES_REGISTRY[valid_objects[goal_indx], colors[goal_indx]]
        for i in range(params.num_distractors+1):
            if i is params.num_distractors:
                grid = grid.at[coords[i][0], coords[i][1]].set(goal_tile)
            else:
                grid = grid.at[coords[i][0], coords[i][1]].set(distractor_tiles[i])
            mask = mask.at[coords[i][0], coords[i][1]].set(
                False
            )

        agent_coords = sample_coordinates(keys[4], grid, num=1, mask=mask)[0]
        
        _goal_encoding = AgentNearGoal(tile=goal_tile).encode()
        _rule_encoding = EmptyRule().encode()[None, ...]
        
        agent = AgentState(position=agent_coords, direction=sample_direction(keys[5]))
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
    