from typing import List, Tuple, ClassVar, Optional
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import chex
import numpy as np
import gymnasium as gym
from xminigrid.types import TimeStep, StepType, EnvCarryT
from xminigrid.environment import EnvParamsT, EnvParams, Environment
from xminigrid.wrappers import Wrapper


@chex.dataclass
class TimestepInfo:
    timestep: TimeStep
    info: dict
    init_key: PRNGKey


class BAMDPWrapper(Wrapper):

    def __init__(
        self,
        env,
        env_params: EnvParamsT,
        steps_per_rollout: int = 15,
        num_episodes_per_rollout: int = 4,
    ):
        """Wrapper, creates a multi-episode (BA)MDP around a one-episode MDP.

        Automatically deals with - horizons H in the MDP vs horizons H+ in the
        BAMDP, - resetting the tasks - adding the done info to the state (might be
        needed to make states markov)
        """
        super().__init__(env)
        self.env = env
        self.env_params = env_params
        self.steps_per_rollout = steps_per_rollout

        # calculate horizon length H^+
        try:
            self.bamdp_horizon = env_params.max_episode_steps * num_episodes_per_rollout
        except:
            self.bamdp_horizon = steps_per_rollout * num_episodes_per_rollout

    def reset(self, env_params: EnvParamsT, rng: PRNGKey, **kwargs):
        timestep = self.env.reset(env_params, rng, **kwargs)

        xtimestep = TimestepInfo(
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

    def __auto_reset(
        self, env_params: EnvParamsT, timestep: TimeStep, key: PRNGKey, **kwargs
    ):
        # key, _ = jax.random.split(timestep.state.key)
        # always reset to the same initial state after a trial is complete
        # TODO: not sure if this is always correct
        reset_timestep = self._env.reset(env_params, key, **kwargs)

        timestep = timestep.replace(
            state=reset_timestep.state,
            observation=reset_timestep.observation,
        )
        return timestep

    def step(
        self,
        env_params: EnvParamsT,
        xtimestep: TimestepInfo,
        action: jax.Array,
        **kwargs
    ):
        timestep = self.env.step(env_params, xtimestep.timestep, action)

        # ignore environment done
        info = xtimestep.info
        info["step_count"] = info["step_count"] + 1
        steps = info["step_count"]
        done_mdp = jnp.where(steps >= self.max_episode_steps, 1, 0)
        info["done_mdp"] = done_mdp

        # when the MDP is done, we reset back to initial timestep, but keep the same task
        timestep = jax.lax.cond(
            done_mdp,
            lambda: self.__auto_reset(
                env_params, timestep, xtimestep.init_key, **kwargs
            ),
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
        done_bamdp = jnp.where(info["step_count_bamdp"] == self.bamdp_horizon, 1, 0)
        info["done_bamdp"] = done_bamdp
        info["done"] = done_bamdp

        # finish the episode
        timestep = timestep.replace(
            step_type=jnp.where(done_bamdp, StepType.LAST, StepType.MID)
        )
        xtimestep = TimestepInfo(
            timestep=timestep, info=info, init_key=xtimestep.init_key
        )
        return xtimestep

    @property
    def observation_space(self):
        try:
            return self.env.observation_space(self.env_params)
        except:
            return gym.spaces.Box(
                shape=self.env.observation_shape(self.env_params), low=0, high=1
            )

    @property
    def action_space(self):
        try:
            return self.env.action_space(self.env_params)
        except:
            return gym.spaces.Discrete(self.env.num_actions(self.env_params))

    @property
    def max_episode_steps(self):
        try:
            return self.env_params.max_episode_steps
        except:
            return self.steps_per_rollout
