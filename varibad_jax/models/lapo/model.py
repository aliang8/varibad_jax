import dataclasses
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
from varibad_jax.models.common import ImageEncoder, ImageDecoder
from ml_collections.config_dict import ConfigDict


@dataclasses.dataclass
class LatentFDM(hk.Module):
    def __init__(
        self,
        config: ConfigDict,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs
    ):
        """Forward Dynamics Model in LAPO, they call this a World Model

        Args:
          pass
        """
        super().__init__(name="LatentFDM")

        init_kwargs = dict(w_init=w_init, b_init=b_init)

        self.image_obs = config.image_obs

        if self.image_obs:
            self.state_embed = ImageEncoder(
                **config.image_encoder_config, **init_kwargs
            )
            self.state_decoder = ImageDecoder(
                **config.image_decoder_config, **init_kwargs
            )
        else:
            self.embedding_dim = config.embedding_dim
            self.state_embed = hk.Linear(
                self.embedding_dim, name="state_embed", **init_kwargs
            )

    def __call__(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        is_training: bool = True,
    ):
        """FDM takes the state and latent action and predicts the next state

        Note that the LAPO world model uses a U-Net style architecture for image input

        Also LAPO does a weird thing where they merge the TC dims

        Input:
            states: (B, T, D) or (B, T, C, H, W)
            actions: (B, T)

        Output:
            next_state_pred: (B, T, D) or (B, T, C, H, W)
        """

        if self.image_obs:
            state_embed = self.state_embed(states, is_training=is_training)
        else:
            state_embed = self.state_embed(states)

        # embed the actions and concatenate with the state embedding
        action_embed = self.latent_embed(actions)
        decoder_input = jnp.concatenate([state_embed, action_embed], axis=-1)

        # decoder and predict the next state
        if self.image_obs:
            next_state_pred = self.state_decoder(decoder_input, is_training=is_training)

        return next_state_pred


@dataclasses.dataclass
class LatentActionIDM(hk.Module):
    def __init__(
        self,
        config: ConfigDict,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs
    ):
        """Inverse Dynamics Model in LAPO with Quantization

        Args:
          pass
        """
        super().__init__(name="LatentActionIDM")

        init_kwargs = dict(w_init=w_init, b_init=b_init)

        self.image_obs = config.image_obs

        if self.image_obs:
            self.state_embed = ImageEncoder(
                **config.image_encoder_config, **init_kwargs
            )
        else:
            self.embedding_dim = config.embedding_dim
            self.state_embed = hk.Linear(
                self.embedding_dim, name="state_embed", **init_kwargs
            )

        # see VQVAE in https://arxiv.org/abs/1711.00937
        # the EMA version uses an exponential moving average of the embedding vectors
        self.vq = hk.nets.VectorQuantizerEMA(
            embedding_dim=config.code_dim,
            num_embeddings=config.num_codes,
            commitment_cost=config.beta,
            decay=config.ema_decay,
            name="VQEMA",
        )

    def __call__(
        self, states: jnp.ndarray, next_states: jnp.ndarray, is_training: bool = True
    ):
        """IDM takes the state and next state and predicts the action

        states: (B, T, D) or (B, T, C, H, W)
        next_states: (B, T, D) or (B, T, C, H, W)
        """
        jax.debug.breakpoint()
        if self.image_obs:
            state_embeds = self.state_embed(states, is_training=is_training)
            next_state_embeds = self.state_embed(next_states, is_training=is_training)
        else:
            state_embeds = self.state_embed(states)
            next_state_embeds = self.state_embed(next_states)

        policy_input = jnp.concatenate([state_embeds, next_state_embeds], axis=-1)
        policy_input = nn.gelu(policy_input)

        # predict latent actions
        latent_actions = self.policy_head(policy_input)

        # compute quantized latent actions
        quantize, loss, perplexity, encodings, encoding_indices = self.vq(
            latent_actions, is_training
        )

        model_outputs = LAMOutputs(
            z_e=encodings,
            z_q=quantize,
            obs_pred=None,
            codebook_loss=loss,
            encoding_indices=encoding_indices,
        )

        return model_outputs
