from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import chex
import einops
import haiku as hk
import jax.numpy as jnp
from ml_collections.config_dict import ConfigDict, FrozenConfigDict

from varibad_jax.models.decoder import Decoder
from varibad_jax.models.lstm_encoder import LSTMTrajectoryEncoder
from varibad_jax.models.transformer_encoder import SARTransformerEncoder
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
        w_init = hk.initializers.VarianceScaling(scale=2.0)
        b_init = hk.initializers.Constant(0.0)
        init_kwargs = dict(w_init=w_init, b_init=b_init)

        if self.config.encoder == "lstm":
            encoder_cls = LSTMTrajectoryEncoder
            encoder_kwargs = dict(
                image_obs=config.image_obs,
                embedding_dim=config.embedding_dim,
                lstm_hidden_size=config.lstm_hidden_size,
                batch_first=False,
            )
        elif self.config.encoder == "transformer":
            encoder_cls = SARTransformerEncoder
            encoder_kwargs = dict(
                image_obs=config.image_obs,
                embedding_dim=config.embedding_dim,
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                attn_size=config.attn_size,
                num_layers=config.num_layers,
                dropout_rate=config.dropout_rate,
                widening_factor=config.widening_factor,
                max_timesteps=config.max_timesteps,
            )

        self.encoder = encoder_cls(**encoder_kwargs)

        self.latent_mean = hk.Linear(
            self.config.latent_dim, name="latent_mean", **init_kwargs
        )
        self.latent_logvar = hk.Linear(
            self.config.latent_dim, name="latent_logvar", **init_kwargs
        )

        self.reward_decoder = Decoder(
            input_action=config.input_action,
            input_prev_state=config.input_prev_state,
            embedding_dim=config.embedding_dim,
            layer_sizes=list(config.rew_decoder_layers),
        )

    def get_prior(self, batch_size: int):
        if self.config.encoder == "lstm":
            hidden_state = self.encoder.recurrent.initial_state(batch_size)
        elif self.config.encoder == "transformer":
            hidden_state = jnp.zeros((batch_size, self.config.hidden_dim))

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

    def encode(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        hidden_state: jnp.ndarray = None,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
        **kwargs
    ):
        hidden_state = self.encoder(
            states=states,
            actions=actions,
            rewards=rewards,
            hidden_state=hidden_state,
            mask=mask,
            deterministic=deterministic,
        )

        # predict distribution for latent variable for each timestep
        latent_mean = self.latent_mean(hidden_state)
        latent_logvar = self.latent_logvar(hidden_state)
        # clamp logvar
        latent_logvar = jnp.clip(latent_logvar, a_min=-10, a_max=10)

        latent_dist = tfd.Normal(loc=latent_mean, scale=jnp.exp(latent_logvar))
        latent_sample = latent_dist.sample(seed=hk.next_rng_key())

        # if just a single timestep, remove the time dimension
        if latent_mean.shape[0] == 1:
            hidden_state = hidden_state[0]
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
