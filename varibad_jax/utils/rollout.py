import jax
import chex
import jax.numpy as jnp
from xminigrid.environment import Environment, EnvParamsT
from flax.training.train_state import TrainState
from typing import Callable
import numpy as np


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
    steps_per_rollout: int = 1,
    num_eval_rollouts: int = 1,
) -> RolloutStats:

    # @jax.jit
    # def _cond_fn(carry):
    #     rng, stats, timestep, prev_action, prev_reward, latent, hidden_state = carry
    #     return jnp.less(jnp.sum(stats.episodes), 1)

    # @jax.jit
    # def _body_fn(carry):
    #     rng, stats, prev_obs, prev_action, prev_reward, latent, hidden_state = carry

    stats = RolloutStats(
        reward=jnp.zeros((config.env.num_processes, 1)),
        length=jnp.zeros((config.env.num_processes, 1)),
    )

    init_state = env.reset()
    init_state = init_state.astype(jnp.float32)
    prev_obs = init_state
    prev_action = jnp.zeros((config.env.num_processes, action_dim))
    prev_reward = jnp.zeros((config.env.num_processes, 1))

    rng, prior_rng = jax.random.split(rng)
    prior_outputs = get_prior(
        ts_vae.params, prior_rng, batch_size=config.env.num_processes
    )
    latent_mean = prior_outputs.latent_mean
    latent_logvar = prior_outputs.latent_logvar
    latent = jnp.concatenate([latent_mean, latent_logvar], axis=-1)
    hidden_state = prior_outputs.hidden_state

    for _ in range(steps_per_rollout):
        rng, policy_rng, encoder_rng = jax.random.split(rng, 3)
        policy_output = ts_policy.apply_fn(
            ts_policy.params,
            policy_rng,
            env_state=prev_obs,
            latent=latent,
        )
        action = policy_output.action
        next_obs, reward, done, info = env.step(action)
        next_obs = next_obs.astype(jnp.float32)

        # add extra dimension for batch
        if len(next_obs.shape) == 1:
            next_obs = next_obs[jnp.newaxis]

        if len(action.shape) == 1:
            action = action[..., jnp.newaxis].astype(jnp.float32)

        if len(reward.shape) == 1:
            reward = reward[..., jnp.newaxis]

        if len(done.shape) == 1:
            done = done[..., jnp.newaxis]

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
            reward=stats.reward + reward,
            length=stats.length + 1,
        )

    metrics = {
        "return": np.array(jnp.sum(stats.reward)),
        "avg_length": np.array(jnp.mean(stats.length)),
    }
    return metrics
    # init_carry = (
    #     rng,
    #     RolloutStats(
    #         reward=jnp.zeros((config.env.num_processes, 1)),
    #         length=jnp.zeros((config.env.num_processes, 1)),
    #         episodes=jnp.zeros((config.env.num_processes, 1)),
    #     ),
    #     init_state,
    #     prev_action,
    #     prev_reward,
    #     latent,
    #     hidden_state,
    # )

    # final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    # return final_carry[1]
