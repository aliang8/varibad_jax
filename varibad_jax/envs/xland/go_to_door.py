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
    random_color: bool = struct.field(pytree_node=False, default=False)
    new_color: bool = struct.field(pytree_node=False, default=False)
    shift_doors: bool = struct.field(pytree_node=False, default=False)


class GoToDoor(CustomXLandMiniGrid):
    def default_params(self, **kwargs) -> EnvParams:
        params = EnvParams(height=9, width=9)
        params = params.replace(**kwargs)

        if params.max_steps is None:
            # formula directly taken from MiniGrid
            params = params.replace(max_steps=4 * (params.height * params.width))
        return params

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State[EnvCarry]:
        key, _key = jax.random.split(key)
        keys = jax.random.split(_key, num=5)

        # pick four random color for the tiles without replacement
        # colors = jax.random.permutation(keys[0], jnp.arange(1, NUM_COLORS))[:4]

        # select four items
        colors = jnp.array([Colors.RED, Colors.GREEN, Colors.BLUE, Colors.YELLOW])

        valid_colors = jnp.array(
            [
                Colors.RED,
                Colors.GREEN,
                Colors.BLUE,
                Colors.YELLOW,
                Colors.ORANGE,
                Colors.PURPLE,
            ]
        )
        if params.random_color:
            colors = jax.random.choice(keys[0], valid_colors, shape=(4,), replace=False)

        # pick one of the tiles to be the goal
        goal_indx = jax.random.randint(keys[2], shape=(), minval=0, maxval=len(colors))

        tile = Tiles.DOOR_LOCKED
        colored_tiles = []

        goal_tile = TILES_REGISTRY[tile, colors[goal_indx]]

        if params.new_color:

            # new_color = jax.random.randint(
            #     keys[0], shape=(1,), minval=1, maxval=NUM_COLORS
            # )
            new_color = Colors.PINK
            # pick a random color to replace
            replace_indx = jax.random.randint(
                keys[1], shape=(), minval=0, maxval=len(colors)
            )
            colors = colors.at[replace_indx].set(new_color)
            goal_tile = TILES_REGISTRY[tile, new_color]

        for _, color in enumerate(colors):
            colored_tiles.append(TILES_REGISTRY[tile, color])

        grid = room(params.height, params.width)

        # pick four numbers between 1 and params.height - 1
        # door_positions = jax.random.randint(
        #     keys[1], shape=(4,), minval=1, maxval=params.height - 1
        # )
        if params.shift_doors:
            door_positions = [
                int(params.height // 2) + 1,
                int(params.height // 2) + 1,
                int(params.width // 2) - 1,
                int(params.width // 2) - 1,
            ]
        else:
            door_positions = [
                int(params.height // 2),
                int(params.height // 2),
                int(params.width // 2),
                int(params.width // 2),
            ]

        grid = grid.at[0, door_positions[0]].set(colored_tiles[0])
        grid = grid.at[door_positions[1], -1].set(colored_tiles[1])
        grid = grid.at[-1, door_positions[2]].set(colored_tiles[2])
        grid = grid.at[door_positions[3], 0].set(colored_tiles[3])

        adj_tile = jax.lax.switch(
            goal_indx,
            (
                lambda: [1, door_positions[0]],
                lambda: [door_positions[1], params.height - 2],
                lambda: [params.width - 2, door_positions[2]],
                lambda: [door_positions[3], 1],
            ),
        )

        # also spawn a ball in the environment to indicate which door is the goal
        ball = TILES_REGISTRY[Tiles.BALL, goal_tile[-1]]
        mask = jnp.ones((grid.shape[0], grid.shape[1]), dtype=jnp.bool_)
        mask = mask.at[adj_tile[0], adj_tile[1]].set(
            False
        )  # the ball should not spawn next to the goal door

        _goal_encoding = AgentNearGoal(tile=goal_tile).encode()
        _rule_encoding = EmptyRule().encode()[None, ...]

        mask = mask.at[int(params.height // 2), int(params.width // 2)].set(False)

        ball_coords, agent_coords = sample_coordinates(keys[3], grid, num=2, mask=mask)
        # grid = grid.at[ball_coords[0], ball_coords[1]].set(ball)
        grid = grid.at[int(params.height // 2), int(params.width // 2)].set(ball)

        agent = AgentState(position=agent_coords, direction=sample_direction(keys[4]))
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
