from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import chex
import einops
import haiku as hk
import jax.numpy as jnp
from ml_collections.config_dict import ConfigDict, FrozenConfigDict

from varibad_jax.models.varibad.decoder import RewardDecoder, TaskDecoder
from varibad_jax.models.varibad.lstm_encoder import LSTMTrajectoryEncoder
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
    task_pred: Optional[jnp.ndarray] = None
    termination_pred: Optional[jnp.ndarray] = None


class VaribadVAE(hk.Module):
    """VariBAD."""

    def __init__(self, config: FrozenConfigDict):
        super().__init__(name="VariBADVAE")
        self.config = config
        w_init = hk.initializers.VarianceScaling(scale=2.0)
        b_init = hk.initializers.Constant(0.0)
        init_kwargs = dict(w_init=w_init, b_init=b_init)

        if self.config.encoder.name == "lstm":
            encoder_cls = LSTMTrajectoryEncoder
        elif self.config.encoder.name == "transformer":
            encoder_cls = SARTransformerEncoder

        self.encoder = encoder_cls(**self.config.encoder, **init_kwargs)

        self.latent_mean = hk.Linear(
            self.config.latent_dim, name="latent_mean", **init_kwargs
        )
        self.latent_logvar = hk.Linear(
            self.config.latent_dim, name="latent_logvar", **init_kwargs
        )

        if self.config.decode_rewards:
            self.reward_decoder = RewardDecoder(**self.config.decoder)

        if self.config.decode_tasks:
            self.task_decoder = TaskDecoder(**self.config.decoder)
        
        if self.config.decode_states:
            # TODO: fix this
            self.state_decoder = hk.Linear(
                self.config.state_dim, name="state_decoder", **init_kwargs
            )

    def get_prior(self, batch_size: int):
        if self.config.encoder.name == "lstm":
            hidden_state = self.encoder.recurrent.initial_state(batch_size)
        elif self.config.encoder.name == "transformer":
            hidden_state = jnp.zeros((batch_size, self.config.encoder.hidden_dim))

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
        is_training: bool = True,
        **kwargs
    ):
        hidden_state = self.encoder(
            states=states,
            actions=actions,
            rewards=rewards,
            hidden_state=hidden_state,
            mask=mask,
            is_training=is_training,
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
        is_training: bool = True,
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
        if self.config.decode_rewards:
            rew_pred = self.reward_decoder(
                latents=latent_samples,
                prev_states=prev_states,
                next_states=next_states,
                actions=actions,
                is_training=is_training,
            )
        else:
            rew_pred = None

        # Reconstruct state
        # p(s_t+1 | s_t, a_t, m)
        if self.config.decode_states:
            state_pred = self.state_decoder(
                latents=latent_samples, states=prev_states, actions=actions
            )
        else:
            state_pred = None

        if self.config.decode_tasks:
            task_pred = self.task_decoder(latents=latent_samples)
        else:
            task_pred = None

        decode_outputs = DecodeOutputs(
            rew_pred=rew_pred,
            state_pred=state_pred,
            task_pred=task_pred
        )
        return decode_outputs
