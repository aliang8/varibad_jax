import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
import numpy as np
from varibad_jax.models.varibad_vae import VaribadVAE


@chex.dataclass
class Batch:
    prev_obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray


@hk.transform
def encode_trajectory(config: FrozenConfigDict, **kwargs):
    model = VaribadVAE(config=config)
    return model.encode(**kwargs)


@hk.transform
def get_prior(config: FrozenConfigDict, **kwargs):
    model = VaribadVAE(config=config)
    return model.get_prior(**kwargs)


@hk.transform
def decode(config: FrozenConfigDict, **kwargs):
    model = VaribadVAE(config=config)
    return model.decode(**kwargs)


def init_params(
    config: FrozenConfigDict, rng_key: PRNGKey, state_dim: int, action_dim: int
):
    t = 2
    bs = 2
    dummy_states = np.zeros((t, bs, state_dim))
    dummy_actions = np.zeros((t, bs, action_dim))
    dummy_rewards = np.zeros((t, bs, 1))
    dummy_hs = np.zeros((bs, config.lstm_hidden_size))

    dummy_latents = np.zeros((t, bs, config.latent_dim))

    encoder_key, decoder_key = jax.random.split(rng_key)

    encoder_params = encode_trajectory.init(
        encoder_key,
        config=config,
        **dict(
            states=dummy_states,
            actions=dummy_actions,
            rewards=dummy_rewards,
            hidden_state=dummy_hs,
        )
    )
    decoder_params = decode.init(
        decoder_key,
        config=config,
        **dict(
            latent_samples=dummy_latents,
            prev_states=dummy_states,
            next_states=dummy_states,
            actions=dummy_actions,
        )
    )
    encoder_params.update(decoder_params)
    return encoder_params
