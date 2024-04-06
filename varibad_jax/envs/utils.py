import gymnasium as gym
import numpy as np
import jax
import xminigrid
from typing import ClassVar, Optional

from varibad_jax.envs.gridworld_jax import GridNavi
from varibad_jax.envs.wrappers import BAMDPWrapper

from xminigrid.benchmarks import Benchmark, load_benchmark, load_benchmark_from_path
from xminigrid.rendering.text_render import print_ruleset


def make_envs(
    env_name: str,
    env_id: str,
    seed: int = 0,
    num_envs: int = 1,
    num_episodes_per_rollout: int = 4,
    benchmark_path: str = "",
    ruleset_id: int = 0,
    env_kwargs=dict(),
    training: bool = True,
    **kwargs,
):
    if training == False:
        seed += 1000

    # setup environment
    if env_name == "gridworld":
        env = GridNavi(**env_kwargs)
        possible_goals = env.possible_goals(env_kwargs["grid_size"])
        env_params = env.default_params(possible_goals=possible_goals, **env_kwargs)
    elif env_name == "xland":
        env, env_params = xminigrid.make(env_id, **env_kwargs)

        # either we define the ruleset manually here OR load from benchmark
        loaded_benchmark = load_benchmark_from_path(benchmark_path)
        num_rulesets = loaded_benchmark.num_rulesets()
        assert ruleset_id < num_rulesets, f"ruleset_id {ruleset_id} not in benchmark"

        ruleset = loaded_benchmark.get_ruleset(ruleset_id=ruleset_id)
        print_ruleset(ruleset)
        env_params = env_params.replace(ruleset=ruleset)

    if num_episodes_per_rollout > 1:
        env = BAMDPWrapper(
            env,
            env_params=env_params,
            num_episodes_per_rollout=num_episodes_per_rollout,
        )

    return env, env_params
