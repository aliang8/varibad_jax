import chex
import numpy as np
import haiku as hk
from typing import Any
import jax.numpy as jnp
from jax.random import PRNGKey
from ml_collections.config_dict import ConfigDict
from varibad_jax.agents.ppo.actor_critic import ActorCritic, PolicyOutput
import gymnasium as gym


@hk.transform
def policy_fn(
    config: ConfigDict,
    is_continuous,
    action_dim,
    env_state,
    latent: jnp.ndarray = None,
    task: jnp.ndarray = None,
):
    policy = ActorCritic(config, is_continuous, action_dim)
    policy_output = policy(state=env_state, latent=latent, task=task)
    return policy_output


def init_params(
    config: ConfigDict,
    rng_key: PRNGKey,
    observation_space: gym.spaces.Space,
    latent_dim: int,
    action_space: gym.spaces.Space,
    task_dim: int = 0,
):
    t = 2
    bs = 2
    dummy_states = np.zeros((t, bs, *observation_space.shape), dtype=np.float32)
    if config.pass_latent_to_policy:
        dummy_latents = np.zeros((t, bs, latent_dim))
    else:
        dummy_latents = None

    if config.pass_task_to_policy:
        dummy_tasks = np.zeros((t, bs, task_dim))
    else:
        dummy_tasks = None

    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
    else:
        action_dim = action_space.shape[0]

    params = policy_fn.init(
        rng=rng_key,
        config=config,
        is_continuous=not isinstance(action_space, gym.spaces.Discrete),
        action_dim=action_dim,
        env_state=dummy_states,
        latent=dummy_latents,
        task=dummy_tasks,
    )
    return params
