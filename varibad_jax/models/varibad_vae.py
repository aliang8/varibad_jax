from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import chex
import einops
import haiku as hk
import jax.numpy as jnp
from ml_collections.config_dict import ConfigDict, FrozenConfigDict

from varibad_jax.models.decoder import Decoder
from varibad_jax.models.lstm_encoder import LSTMTrajectoryEncoder


@chex.dataclass
class DecodeOutputs:
    rew_pred: jnp.ndarray
    state_pred: Optional[jnp.ndarray] = None
    termination_pred: Optional[jnp.ndarray] = None


class VaribadVAE(hk.Module):
    """VariBAD."""

    def __init__(self, config: FrozenConfigDict):
        super().__init__(name="VariBADVAE")
        self.config = config

        encoder_cls = LSTMTrajectoryEncoder

        encoder_kwargs = dict(
            embedding_dim=config.embedding_dim,
            latent_dim=config.latent_dim,
            lstm_hidden_size=config.lstm_hidden_size,
        )

        self.encoder = encoder_cls(**encoder_kwargs)

        self.reward_decoder = Decoder(
            input_action=config.input_action,
            input_prev_state=config.input_prev_state,
            embedding_dim=config.embedding_dim,
            layer_sizes=list(config.rew_decoder_layers),
        )

    def encode(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        hidden_state: jnp.ndarray = None,
    ):
        return self.encoder(
            states=states, actions=actions, rewards=rewards, hidden_state=hidden_state
        )

    def decode(
        self,
        latent_samples: jnp.ndarray,
        prev_states: Optional[jnp.ndarray] = None,
        next_states: Optional[jnp.ndarray] = None,
        actions: Optional[jnp.ndarray] = None,
    ):
        """Decode

        Args:
          prev_states: [T, B, D_state] next states: [T, B, D_state]
          actions: [T, B, 1]
          latent_samples: [T, B, D_latent]
          long_term_belief: [B, 2*latent_dim]

        Returns:
          decode_outputs: DecodeOutputs
        """
        rew_pred = self.reward_decoder(
            latents=latent_samples,
            prev_states=prev_states,
            next_states=next_states,
            actions=actions,
        )

        # Reconstruct state
        # p(s_t+1 | s_t, a_t, m)
        # state_pred = self.state_decoder(
        #     latents=latent_samples, states=prev_states, actions=actions
        # )

        decode_outputs = DecodeOutputs(
            rew_pred=rew_pred,
            state_pred=None,
        )
        return decode_outputs
