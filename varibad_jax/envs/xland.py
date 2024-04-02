import jax
import jax.numpy as jnp
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper, Wrapper
from xminigrid.environment import Environment, EnvParamsT
import gymnasium as gym
import numpy as np
import jax
import chex
import xminigrid
from xminigrid.types import TimeStep
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
    max_tiles = 1000

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
    save_bz2_pickle(concat_rulesets, benchmark_path, protocol=-1)


def make_envs(
    env_id: str,
    seed: int = 0,
    num_envs: int = 1,
    num_episodes_per_rollout: int = 4,
    benchmark_path: str = "",
    ruleset_id: int = 0,
    env_kwargs=dict(),
    training: bool = True,
):
    # setup environment
    env, env_params = xminigrid.make(env_id, **env_kwargs)
    # env = GymAutoResetWrapper(env)

    # either we define the ruleset manually here OR load from benchmark
    loaded_benchmark = load_benchmark_from_path(benchmark_path)
    num_rulesets = loaded_benchmark.num_rulesets()
    assert ruleset_id < num_rulesets, f"ruleset_id {ruleset_id} not in benchmark"

    ruleset = loaded_benchmark.get_ruleset(ruleset_id=ruleset_id)
    print_ruleset(ruleset)
    env_params = env_params.replace(ruleset=ruleset)

    env = BAMDPWrapper(
        env, env_params, num_episodes_per_rollout=num_episodes_per_rollout
    )
    if num_envs > 1 and training:
        env = VmapWrapper(env, num_envs=num_envs)

    if training:
        env = GymWrapper(env, env_params, seed)
    return env


@chex.dataclass
class XLandTimestep:
    timestep: TimeStep
    info: dict
    init_key: PRNGKey


class BAMDPWrapper(Wrapper):

    def __init__(self, env, env_params: EnvParamsT, num_episodes_per_rollout: int):
        """Wrapper, creates a multi-episode (BA)MDP around a one-episode MDP.

        Automatically deals with - horizons H in the MDP vs horizons H+ in the
        BAMDP, - resetting the tasks - adding the done info to the state (might be
        needed to make states markov)
        """
        super().__init__(env)
        self.env = env
        self._env_params = env_params

        # calculate horizon length H^+
        self.num_episodes = num_episodes_per_rollout
        self.max_episode_steps = 15  # self.env.time_limit(env_params)
        self.episode_length = self.max_episode_steps
        self.horizon = self.episode_length * self.num_episodes

    def reset(self, env_params: EnvParamsT, rng: PRNGKey):
        timestep = self.env.reset(env_params, rng)

        xtimestep = XLandTimestep(
            timestep=timestep,
            info=dict(
                step_count=0,
                step_count_bamdp=0,
                done_mdp=0,
                done_bamdp=0,
                done=0,
            ),
            init_key=rng,
        )
        return xtimestep

    def __auto_reset(self, env_params: EnvParamsT, timestep: TimeStep, key: PRNGKey):
        # key, _ = jax.random.split(timestep.state.key)
        # always reset to the same initial state after a trial is complete
        # TODO: not sure if this is always correct
        reset_timestep = self._env.reset(env_params, key)

        timestep = timestep.replace(
            state=reset_timestep.state,
            observation=reset_timestep.observation,
        )
        return timestep

    def step(self, env_params: EnvParamsT, xtimestep: XLandTimestep, action: jax.Array):
        timestep = self.env.step(env_params, xtimestep.timestep, action)

        # ignore environment done
        info = xtimestep.info
        info["step_count"] = info["step_count"] + 1
        steps = info["step_count"]
        done_mdp = jnp.where(steps >= self.episode_length, 1, 0)
        info["done_mdp"] = done_mdp

        # when the MDP is done, we reset back to initial timestep, but keep the same task
        timestep = jax.lax.cond(
            done_mdp,
            lambda: self.__auto_reset(env_params, timestep, xtimestep.init_key),
            lambda: timestep,
        )

        # also reset step count
        info["step_count"] = jnp.where(
            done_mdp,
            jnp.zeros_like(info["step_count"]),
            info["step_count"],
        )

        # check if the entire rollout is complete
        info["step_count_bamdp"] = info["step_count_bamdp"] + 1
        done_bamdp = jnp.where(info["step_count_bamdp"] == self.horizon, 1, 0)
        info["done_bamdp"] = done_bamdp
        info["done"] = done_bamdp

        # finish the episode
        timestep = timestep.replace(
            step_type=jnp.where(done_bamdp, StepType.LAST, StepType.FIRST)
        )

        # info["truncation"] = jnp.where(
        #     info["step_count_bamdp"] >= self.horizon, 1 - done_bamdp, zero
        # )
        xtimestep = XLandTimestep(
            timestep=timestep, info=info, init_key=xtimestep.init_key
        )
        return xtimestep

    @property
    def env_params(self):
        return self._env_params


class VmapWrapper(Wrapper):
    """Vectorizes JAX env."""

    def __init__(self, env: Environment, num_envs: Optional[int] = None):
        super().__init__(env)
        self._env = env
        self.num_envs = num_envs

    def reset(self, env_params: EnvParamsT, rng: jax.Array):
        if self.num_envs is not None:
            rng = jax.random.split(rng, self.num_envs)

        return jax.vmap(self._env.reset, in_axes=(None, 0))(env_params, rng)

    def step(self, env_params: EnvParamsT, xtimestep: XLandTimestep, action: jax.Array):
        return jax.vmap(self._env.step, in_axes=(None, 0, 0))(
            env_params, xtimestep, action
        )

    @property
    def max_episode_steps(self):
        return self._env.max_episode_steps

    @property
    def env_params(self):
        return self._env.env_params


class GymWrapper(Wrapper):
    """A wrapper that converts JAX Env to one that follows Gym API."""

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env, env_params: EnvParamsT, seed: int = 0):
        self._env = env
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
        }
        self.env_params = env_params
        self.seed(seed)
        self._xtimestep = None
        self.observation_shape = self._env.observation_shape(env_params)
        self.num_actions = self._env.num_actions(env_params)

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.observation_shape, dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.task_dim = 0  # TODO: fix this

        def reset(key):
            key1, key2 = jax.random.split(key)
            xtimestep = self._env.reset(self.env_params, key2)
            timestep = xtimestep.timestep
            return xtimestep, timestep.observation, key1

        self._reset = jax.jit(reset)

        def step(xtimestep, action):
            xtimestep = self._env.step(self.env_params, xtimestep, action)

            timestep = xtimestep.timestep
            return (
                xtimestep,
                timestep.observation,
                timestep.reward,
                timestep.last(),
                timestep.state,
            )

        self._step = jax.jit(step)

    def reset(self):
        self._xtimestep, obs, self._key = self._reset(self._key)
        # We return device arrays for pytorch users.
        return obs

    def step(self, action):
        self._xtimestep, obs, reward, done, info = self._step(self._xtimestep, action)
        # We return device arrays for pytorch users.
        return obs, reward, done, info

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

    @property
    def max_episode_steps(self):
        return self._env.max_episode_steps

    def close(self):
        self._env.close()
