import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from ml_collections.config_dict import ConfigDict, FrozenConfigDict
import numpy as np
from varibad_jax.models.varibad.vae import VaribadVAE


@chex.dataclass
class Batch:
    prev_obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray


@hk.transform_with_state
def encode_trajectory(config: FrozenConfigDict, **kwargs):
    model = VaribadVAE(config=config)
    return model.encode(**kwargs)


@hk.transform
def get_prior(config: FrozenConfigDict, **kwargs):
    model = VaribadVAE(config=config)
    return model.get_prior(**kwargs)


@hk.transform_with_state
def decode(config: FrozenConfigDict, **kwargs):
    model = VaribadVAE(config=config)
    return model.decode(**kwargs)


def init_params(
    config: FrozenConfigDict,
    rng_key: PRNGKey,
    observation_space: tuple,
    action_dim: int,
):
    t = 2
    bs = 2
    dummy_states = np.zeros((t, bs, *observation_space.shape), dtype=np.float32)
    dummy_actions = np.zeros((t, bs, action_dim))
    dummy_rewards = np.zeros((t, bs, 1))
    dummy_hs = np.zeros((bs, config.encoder.lstm_hidden_size))
    dummy_latents = np.zeros((t, bs, config.latent_dim))
    if config.encoder == "transformer":
        dummy_mask = np.ones((t, bs))
    else:
        dummy_mask = None

    encoder_key, decoder_key = jax.random.split(rng_key)

    encoder_params, encoder_state = encode_trajectory.init(
        encoder_key,
        config=config,
        **dict(
            states=dummy_states,
            actions=dummy_actions,
            rewards=dummy_rewards,
            hidden_state=dummy_hs,
            mask=dummy_mask,
            is_training=True,
        )
    )
    decoder_params, decoder_state = decode.init(
        decoder_key,
        config=config,
        **dict(
            latent_samples=dummy_latents,
            prev_states=dummy_states,
            next_states=dummy_states,
            actions=dummy_actions,
            is_training=True,
        )
    )
    encoder_params.update(decoder_params)
    encoder_state.update(decoder_state)
    return encoder_params, encoder_state
