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
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
    ):
        super().__init__(name="LSTMTrajectoryEncoder")

        init_kwargs = dict(w_init=w_init, b_init=b_init)

        self.image_obs = image_obs
        self.embedding_dim = embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.batch_first = batch_first

        if self.image_obs:
            self.state_embed = hk.Sequential(
                [
                    hk.Conv2D(
                        16,
                        (2, 2),
                        padding="VALID",
                        w_init=w_init,
                    ),
                    nn.gelu,
                    hk.Conv2D(
                        32,
                        (2, 2),
                        padding="VALID",
                        w_init=w_init,
                    ),
                    nn.gelu,
                    hk.Conv2D(
                        64,
                        (2, 2),
                        padding="VALID",
                        w_init=w_init,
                    ),
                    nn.gelu,
                    hk.Flatten(preserve_dims=1),
                    hk.Linear(
                        self.embedding_dim,
                        w_init=w_init,
                    ),
                ]
            )
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
        T, B, *_ = states.shape
        if self.image_obs:
            # flatten the time and batch dimensions and run through CNN
            states = jnp.reshape(states, (-1, *states.shape[2:]))
            state_embeds = self.state_embed(states)
            # reshape back
            state_embeds = jnp.reshape(state_embeds, (T, B, -1))
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
        # hidden_state, _ = hk.dynamic_unroll(
        #     core=self.recurrent,
        #     input_sequence=encoder_input,
        #     initial_state=initial_state,
        #     time_major=not self.batch_first,
        #     reverse=False,
        #     return_all_states=False,
        #     unroll=1,
        # )
        hidden_state, final_hidden_state = hk.static_unroll(
            core=self.recurrent,
            input_sequence=encoder_input,
            initial_state=initial_state,
            time_major=not self.batch_first,
        )
        return hidden_state
