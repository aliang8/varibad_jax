import gymnasium as gym
import numpy as np
import jax
import xminigrid
from functools import partial
from typing import ClassVar, Optional

import varibad_jax.envs
from varibad_jax.envs.gridworld_jax import GridNavi
from varibad_jax.envs.wrappers import BAMDPWrapper, GymAutoResetWrapper, BasicWrapper

# from procgen import ProcgenEnv
from xminigrid.benchmarks import Benchmark, load_benchmark, load_benchmark_from_path
from xminigrid.rendering.text_render import print_ruleset


# def normalize_return(ep_ret, env_name):
#     """normalizes returns based on URP and expert returns above"""
#     return doy.normalize_into_range(
#         lower=urp_ep_return[env_name],
#         upper=expert_ep_return[env_name],
#         v=ep_ret,
#     )


# taken from: https://github.com/schmidtdominik/LAPO/blob/main/lapo/env_utils.py
def make_procgen_envs(num_envs, env_id, gamma, **kwargs):
    import gym

    envs = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_id,
        num_levels=0,
        start_level=0,
        distribution_mode="easy",
    )

    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    assert isinstance(
        envs.action_space, gym.spaces.discrete.Discrete
    ), "only discrete action space is supported"

    # envs.normalize_return = partial(normalize_return, env_name=env_id)
    return envs


def make_envs(
    env_name: str,
    env_id: str,
    seed: int = 0,
    num_envs: int = 1,
    num_episodes_per_rollout: int = 4,
    steps_per_rollout: int = 15,
    benchmark_path: str = "",
    ruleset_id: int = 0,
    preloaded_benchmark: str = "",
    env_kwargs=dict(),
    training: bool = True,
    **kwargs,
):
    if training == False:
        seed += 1000

    # setup environment
    if env_name == "gridworld":
        env = GridNavi(**env_kwargs)
        env_params = env.default_params(**env_kwargs)
    elif env_name == "xland":
        env_kwargs["max_steps"] = steps_per_rollout
        env, env_params = xminigrid.make(env_id, **env_kwargs)

        if ruleset_id > -1:
            # either we define the ruleset manually here OR load from benchmark
            if preloaded_benchmark:
                loaded_benchmark = xminigrid.load_benchmark(name=preloaded_benchmark)
            else:
                loaded_benchmark = load_benchmark_from_path(benchmark_path)

            num_rulesets = loaded_benchmark.num_rulesets()
            assert (
                ruleset_id < num_rulesets
            ), f"ruleset_id {ruleset_id} not in benchmark"

            ruleset = loaded_benchmark.get_ruleset(ruleset_id=ruleset_id)
            print_ruleset(ruleset)
            env_params = env_params.replace(ruleset=ruleset)

    if training:
        env = GymAutoResetWrapper(env)

    # this still works if we have num_episodes_per_rollout = 1
    if num_episodes_per_rollout > 1:
        env = BAMDPWrapper(
            env,
            env_params=env_params,
            steps_per_rollout=steps_per_rollout,
            num_episodes_per_rollout=num_episodes_per_rollout,
        )
    else:
        env = BasicWrapper(
            env, env_params=env_params, steps_per_rollout=steps_per_rollout
        )

    return env, env_params
