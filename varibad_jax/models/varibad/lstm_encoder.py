from collections import deque
import dataclasses
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import math
import chex
import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from flax.linen.initializers import glorot_normal, orthogonal, zeros_init
from varibad_jax.models.common import ImageEncoder

tfd = tfp.distributions


@dataclasses.dataclass
class LSTMTrajectoryEncoder(hk.Module):
    """Trajectory encoder."""

    def __init__(
        self,
        embedding_dim: int = 8,
        lstm_hidden_size: int = 64,
        batch_first: bool = False,
        image_obs: bool = False,
        image_encoder_config: dict = None,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs
    ):
        super().__init__(name="LSTMTrajectoryEncoder")

        init_kwargs = dict(w_init=w_init, b_init=b_init)

        self.image_obs = image_obs
        self.embedding_dim = embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.batch_first = batch_first

        if self.image_obs:
            self.state_embed = ImageEncoder(**image_encoder_config, **init_kwargs)
        else:
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
        self.recurrent = hk.GRU(self.lstm_hidden_size)

    def __call__(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        hidden_state: jnp.ndarray = None,
        is_training: bool = True,
        **kwargs
    ):
        """Call.

        Args:
            states: [T, B, *obs_shape], during inference this is just [B, state_dim]
            actions: [T, B, action_dim]
            rewards: [T, B, 1]
            hidden_state: [B, hidden_dim]

        Returns:
        """
        if self.image_obs:
            state_embeds = self.state_embed(states, is_training=is_training)
        else:
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
        hidden_state, final_hidden_state = hk.static_unroll(
            core=self.recurrent,
            input_sequence=encoder_input,
            initial_state=initial_state,
            time_major=not self.batch_first,
        )
        return hidden_state
