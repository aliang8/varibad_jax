import jax
import chex
import jax.numpy as jnp
from xminigrid.environment import Environment, EnvParamsT
from flax.training.train_state import TrainState
from typing import Callable
import numpy as np
from xminigrid.experimental.img_obs import _render_obs


@chex.dataclass
class RolloutStats:
    reward: jnp.ndarray
    length: jnp.ndarray


def eval_rollout(
    rng: jax.Array,
    env: Environment,
    config: dict,
    ts_policy: TrainState,
    ts_vae: TrainState,
    get_prior: Callable,
    action_dim: int,
    steps_per_epoch: int,
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

    if visualize:
        img = _render_obs(xtimestep.timestep.state.grid)
        imgs = jnp.zeros((steps_per_epoch, *img.shape))
        imgs = imgs.at[stats.length].set(img)

    # jax.debug.breakpoint()

    def _cond_fn(carry):
        (
            rng,
            stats,
            timestep,
            prev_action,
            prev_reward,
            done,
            latent,
            hidden_state,
            imgs,
        ) = carry
        # while not done
        return jnp.logical_not(done)

    def _body_fn(carry):
        (
            rng,
            stats,
            xtimestep,
            prev_action,
            prev_reward,
            done,
            latent,
            hidden_state,
            imgs,
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

        if visualize:
            img = _render_obs(xtimestep.timestep.state.grid)
            imgs = imgs.at[stats.length + 1].set(img)

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

        return (rng, stats, xtimestep, action, reward, done, latent, hidden_state, imgs)

    init_carry = (
        rng,
        stats,
        xtimestep,
        prev_action,
        prev_reward,
        done,
        latent,
        hidden_state,
        imgs,
    )

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    return final_carry[1], final_carry[-1]
