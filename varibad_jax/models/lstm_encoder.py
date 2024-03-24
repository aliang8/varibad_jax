from collections import deque
import dataclasses
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import chex
import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


@chex.dataclass
class EncodeOutputs:
    latent_mean: jnp.ndarray
    latent_logvar: jnp.ndarray
    latent_dist: jnp.ndarray
    # [T, B, hidden_size]
    hidden_state: jnp.ndarray
    latent_sample: jnp.ndarray


@dataclasses.dataclass
class LSTMTrajectoryEncoder(hk.Module):
    """Trajectory encoder."""

    def __init__(
        self,
        embedding_dim: int = 8,
        latent_dim: int = 5,
        lstm_hidden_size: int = 64,
        batch_first: bool = False,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.01),
    ):
        super().__init__(name="LSTMTrajectoryEncoder")

        init_kwargs = dict(w_init=w_init, b_init=b_init)

        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.batch_first = batch_first

        self.state_embed = hk.Linear(
            self.embedding_dim, name="state_embed", **init_kwargs
        )
        # self.state_embed = hk.Embed(vocab_size=25, embed_dim=self.embedding_dim)
        self.action_embed = hk.Linear(
            self.embedding_dim, name="action_embed", **init_kwargs
        )
        self.reward_embed = hk.Linear(
            self.embedding_dim, name="reward_embed", **init_kwargs
        )

        self.latent_mean = hk.Linear(self.latent_dim, name="latent_mean", **init_kwargs)
        self.latent_logvar = hk.Linear(
            self.latent_dim, name="latent_logvar", **init_kwargs
        )
        self.recurrent = hk.GRU(self.lstm_hidden_size)

    def get_prior(self, batch_size: int):
        hidden_state = self.recurrent.initial_state(batch_size)

        latent_mean = self.latent_mean(hidden_state)
        latent_logvar = self.latent_logvar(hidden_state)
        # clamp logvar
        latent_logvar = jnp.clip(latent_logvar, a_min=-10, a_max=10)
        latent_dist = tfd.Normal(loc=latent_mean, scale=jnp.exp(latent_logvar))
        latent_sample = latent_dist.sample(seed=hk.next_rng_key())

        return EncodeOutputs(
            latent_mean=latent_mean,
            latent_logvar=latent_logvar,
            latent_dist=latent_dist,
            latent_sample=latent_sample,
            hidden_state=hidden_state,
        )

    def __call__(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        hidden_state: jnp.ndarray = None,
    ):
        """Call.

        Args:
            states: [T, B, state_dim], during inference this is just [B, state_dim]
            actions: [T, B, action_dim]
            rewards: [T, B, 1]
            hidden_state: [B, hidden_dim]

        Returns:
            encode_output: EncodeOutputs
        """
        state_embeds = self.state_embed(states)
        state_embeds = nn.gelu(state_embeds)
        action_embeds = self.action_embed(actions)
        action_embeds = nn.gelu(action_embeds)
        reward_embeds = self.reward_embed(rewards)
        reward_embeds = nn.gelu(reward_embeds)

        # concatenate the transition embeddings
        # [T, B, 3*embedding_dim]
        encoder_input = jnp.concatenate(
            (state_embeds, action_embeds, reward_embeds), axis=-1
        )

        # this is for inference time, add time dimension
        if len(encoder_input.shape) == 2:
            encoder_input = jnp.expand_dims(encoder_input, axis=0)

        if self.batch_first:
            B, T = encoder_input.shape[:2]
        else:
            T, B = encoder_input.shape[:2]

        # initial state to LSTM
        if hidden_state is None:
            initial_state = self.recurrent.initial_state(B)
        else:
            initial_state = hidden_state

        # [T, B, D]
        hidden_state, _ = hk.dynamic_unroll(
            core=self.recurrent,
            input_sequence=encoder_input,
            initial_state=initial_state,
            time_major=not self.batch_first,
            reverse=False,
            return_all_states=False,
            unroll=1,
        )

        # import ipdb

        # ipdb.set_trace()

        # predict distribution for latent variable for each timestep
        latent_mean = self.latent_mean(hidden_state)
        latent_logvar = self.latent_logvar(hidden_state)
        # clamp logvar
        latent_logvar = jnp.clip(latent_logvar, a_min=-10, a_max=10)
        latent = jnp.concatenate((latent_mean, latent_logvar), axis=-1)

        latent_dist = tfd.Normal(loc=latent_mean, scale=jnp.exp(latent_logvar))
        latent_sample = latent_dist.sample(seed=hk.next_rng_key())

        # if just a single timestep, remove the time dimension
        if latent.shape[0] == 1:
            latent, hidden_state = latent[0], hidden_state[0]
            latent_mean = latent_mean[0]
            latent_logvar = latent_logvar[0]
            latent_sample = latent_sample[0]

        return EncodeOutputs(
            latent_mean=latent_mean,
            latent_logvar=latent_logvar,
            latent_dist=latent_dist,
            hidden_state=hidden_state,
            latent_sample=latent_sample,
        )
