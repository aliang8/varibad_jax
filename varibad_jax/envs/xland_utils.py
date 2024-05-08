import jax
import jax.numpy as jnp
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper, Wrapper
import gymnasium as gym
import numpy as np
import jax
import chex
import xminigrid

# from xminigrid.types import TimeStep
from typing import ClassVar, Optional
from jax.random import PRNGKey
from xminigrid.core.constants import Tiles, Colors, TILES_REGISTRY
from xminigrid.rendering.rgb_render import render
from xminigrid.core.goals import check_goal, AgentNearGoal
from xminigrid.core.rules import check_rule, AgentNearRule
from xminigrid.types import RuleSet, StepType
from xminigrid.rendering.text_render import print_ruleset
from xminigrid.benchmarks import Benchmark, load_benchmark, load_benchmark_from_path
from xminigrid.core.rules import EmptyRule
from xminigrid.core.grid import pad_along_axis
from xminigrid.benchmarks import save_bz2_pickle

from pathlib import Path


def encode(ruleset):
    flatten_encoding = jnp.concatenate(
        [ruleset["goal"].encode(), *[r.encode() for r in ruleset["rules"]]]
    ).tolist()
    return tuple(flatten_encoding)


def make_benchmark(
    benchmark_path: str,
):
    goal = AgentNearGoal(tile=TILES_REGISTRY[Tiles.SQUARE, Colors.PURPLE])
    rule = AgentNearRule(
        tile=TILES_REGISTRY[Tiles.BALL, Colors.YELLOW],
        prod_tile=TILES_REGISTRY[Tiles.SQUARE, Colors.PURPLE],
    )

    ruleset = {
        "goal": goal,
        "rules": [rule],
        "init_tiles": jnp.array((TILES_REGISTRY[Tiles.BALL, Colors.YELLOW],)),
        "num_rules": len([r for r in [rule] if not isinstance(r, EmptyRule)]),
    }

    rulesets = []
    rulesets.append(
        {
            "goal": ruleset["goal"].encode(),
            "rules": jnp.vstack([r.encode() for r in ruleset["rules"]]),
            "init_tiles": jnp.array(ruleset["init_tiles"], dtype=jnp.uint8),
            "num_rules": jnp.asarray(ruleset["num_rules"], dtype=jnp.uint8),
        }
    )
    max_rules = 1000
    max_tiles = 5

    concat_rulesets = {
        # "generation_config": vars(args),
        "goals": jnp.vstack([r["goal"] for r in rulesets]),
        "rules": jnp.vstack(
            [pad_along_axis(r["rules"], pad_to=max_rules)[None, ...] for r in rulesets]
        ),
        "init_tiles": jnp.vstack(
            [
                pad_along_axis(r["init_tiles"], pad_to=max_tiles)[None, ...]
                for r in rulesets
            ]
        ),
        "num_rules": jnp.vstack([r["num_rules"] for r in rulesets]),
    }
    print(f"Saving benchmark to {benchmark_path}")

    Path(benchmark_path).parent.mkdir(parents=True, exist_ok=True)
    save_bz2_pickle(concat_rulesets, benchmark_path, protocol=-1)


if __name__ == "__main__":
    make_benchmark(
        "/home/anthony/varibad_jax/varibad_jax/envs/xland_benchmarks/test_ruleset.pkl"
    )
