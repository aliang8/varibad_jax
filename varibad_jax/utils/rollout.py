import jax
import chex
import jax.numpy as jnp
from xminigrid.environment import Environment, EnvParamsT
from flax.training.train_state import TrainState
from typing import Callable
import numpy as np
from xminigrid.experimental.img_obs import TILE_W_AGENT_CACHE, TILE_CACHE, TILE_SIZE
from xminigrid.core.constants import NUM_COLORS, TILES_REGISTRY, Tiles


@chex.dataclass
class RolloutStats:
    reward: jnp.ndarray
    length: jnp.ndarray


# rendering with cached tiles
def _render_obs(obs: jax.Array, agent_location) -> jax.Array:
    view_size = obs.shape[0]

    obs_flat_idxs = obs[:, :, 0] * NUM_COLORS + obs[:, :, 1]
    # render all tiles
    rendered_obs = jnp.take(TILE_CACHE, obs_flat_idxs, axis=0)

    # add agent tile
    agent_tile = TILE_W_AGENT_CACHE[obs_flat_idxs[view_size - 1, view_size // 2]]

    import ipdb

    ipdb.set_trace()
    rendered_obs = rendered_obs.at[view_size - 1, view_size // 2].set(agent_tile)
    # [view_size, view_size, tile_size, tile_size, 3] -> [view_size * tile_size, view_size * tile_size, 3]
    rendered_obs = rendered_obs.transpose((0, 2, 1, 3, 4)).reshape(
        view_size * TILE_SIZE, view_size * TILE_SIZE, 3
    )

    return rendered_obs


def eval_rollout(
    rng: jax.Array,
    env: Environment,
    config: dict,
    ts_policy: TrainState,
    ts_vae: TrainState,
    get_prior: Callable,
    action_dim: int,
    steps_per_rollout: int,
    visualize: bool = False,
) -> RolloutStats:

    stats = RolloutStats(reward=0, length=0)

    rng, reset_rng, prior_rng = jax.random.split(rng, 3)
    xtimestep = env.reset(env.env_params, reset_rng)
    prev_action = jnp.zeros((1, action_dim))
    prev_reward = jnp.zeros((1, 1))
    done = False

    prior_outputs = get_prior(ts_vae.params, prior_rng, batch_size=1)
    latent_mean = prior_outputs.latent_mean
    latent_logvar = prior_outputs.latent_logvar
    latent = jnp.concatenate([latent_mean, latent_logvar], axis=-1)
    hidden_state = prior_outputs.hidden_state

    def _step_fn(carry, _):
        (
            rng,
            stats,
            xtimestep,
            prev_action,
            prev_reward,
            done,
            latent,
            hidden_state,
        ) = carry

        rng, policy_rng, encoder_rng = jax.random.split(rng, 3)
        observation = xtimestep.timestep.observation
        observation = observation.astype(jnp.float32)
        observation = observation[jnp.newaxis]

        policy_output = ts_policy.apply_fn(
            ts_policy.params,
            policy_rng,
            env_state=observation,
            latent=latent,
        )
        action = policy_output.action
        xtimestep = env.step(env.env_params, xtimestep, action.squeeze())
        timestep = xtimestep.timestep
        next_obs = timestep.observation
        reward = timestep.reward
        done = timestep.last()
        next_obs = next_obs.astype(jnp.float32)

        # add extra dimension for batch
        next_obs = next_obs[jnp.newaxis]
        action = action.reshape(1, 1).astype(jnp.float32)
        reward = reward.reshape(1, 1)

        # update hidden state
        encode_outputs = ts_vae.apply_fn(
            ts_vae.params,
            encoder_rng,
            states=next_obs,
            actions=action,
            rewards=reward,
            hidden_state=hidden_state,
        )

        hidden_state = encode_outputs.hidden_state
        latent_mean = encode_outputs.latent_mean
        latent_logvar = encode_outputs.latent_logvar
        latent = jnp.concatenate([latent_mean, latent_logvar], axis=-1)

        stats = stats.replace(
            reward=stats.reward + timestep.reward,
            length=stats.length + 1,
        )

        return (
            rng,
            stats,
            xtimestep,
            action,
            reward,
            done,
            latent,
            hidden_state,
        ), timestep

    init_carry = (
        rng,
        stats,
        xtimestep,
        prev_action,
        prev_reward,
        done,
        latent,
        hidden_state,
    )

    carry, transitions = jax.lax.scan(
        _step_fn, init_carry, None, length=steps_per_rollout
    )
    stats = carry[1]
    return stats, transitions
