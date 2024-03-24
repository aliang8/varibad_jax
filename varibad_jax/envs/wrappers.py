from typing import List, Tuple
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from brax.envs import State, Env, Wrapper


class BAMDPWrapper(Wrapper):

    def __init__(self, env, num_episodes_per_rollout: int):
        """Wrapper, creates a multi-episode (BA)MDP around a one-episode MDP.

        Automatically deals with - horizons H in the MDP vs horizons H+ in the
        BAMDP, - resetting the tasks - adding the done info to the state (might be
        needed to make states markov)
        """
        super().__init__(env)

        # calculate horizon length H^+
        self.num_episodes = num_episodes_per_rollout
        self.episode_length = self.env._max_episode_steps
        self.horizon = self.episode_length * self.num_episodes

    def reset(self, rng: PRNGKey):
        state = self.env.reset(rng)

        zeros = jnp.zeros(rng.shape[:-1], dtype=jnp.int32)
        state.info["done_mdp"] = zeros
        state.info["step_count"] = zeros
        state.info["step_count_bamdp"] = zeros
        state.info["done_bamdp"] = zeros
        return state

    def step(self, state: State, action: jax.Array):
        state = self.env.step(state, action)
        # ignore environment done
        one = jnp.ones_like(state.done, dtype=jnp.int32)
        zero = jnp.zeros_like(state.done, dtype=jnp.int32)
        state.info["step_count"] = state.info["step_count"] + 1
        steps = state.info["step_count"]
        done_mdp = jnp.where(steps >= self.episode_length, one, zero)
        state.info["done_mdp"] = done_mdp

        # when the MDP is done, we reset back to initial state, but keep the same task
        init_xy = jnp.array(self.init_state)
        init_obs = self._get_obs(init_xy)
        new_obs = jnp.where(done_mdp, init_obs, state.obs)
        state = state.replace(obs=new_obs)
        xy_coord = jnp.where(done_mdp, init_xy, state.info["xy_coord"])
        state.info["xy_coord"] = xy_coord

        # also reset step count
        state.info["step_count"] = jnp.where(
            done_mdp,
            jnp.zeros_like(state.info["step_count"]),
            state.info["step_count"],
        )

        # check if the entire rollout is complete
        state.info["step_count_bamdp"] = state.info["step_count_bamdp"] + 1
        done_bamdp = jnp.where(
            state.info["step_count_bamdp"] == self.horizon, one, zero
        )
        state.info["done_bamdp"] = done_bamdp
        state = state.replace(done=done_bamdp)

        state.info["truncation"] = jnp.where(
            state.info["step_count_bamdp"] >= self.horizon, 1 - state.done, zero
        )
        return state
