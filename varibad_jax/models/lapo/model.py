import dataclasses
from typing import Any, Callable, List, NamedTuple, Optional, Tuple
from absl import logging
import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
import einops
from ml_collections.config_dict import ConfigDict

from varibad_jax.models.common import ImageEncoder, ImageDecoder
from varibad_jax.models.lapo.helpers import ImpalaCNN


@dataclasses.dataclass
class LatentFDM(hk.Module):
    def __init__(
        self,
        config: ConfigDict,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs,
    ):
        """Forward Dynamics Model in LAPO, they call this a World Model in LAPO
        FDM predicts o_t+1 given o_t-1, o_t and z_t

        Use U-net style architecture following: https://github.com/schmidtdominik/LAPO/blob/main/lapo/models.py

        Args:
          pass
        """
        super().__init__(name="LatentFDM")

        init_kwargs = dict(w_init=w_init, b_init=b_init)
        self.image_obs = config.image_obs

        if self.image_obs:
            # https://asiltureli.github.io/Convolution-Layer-Calculator/
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
        prev_states: jnp.ndarray,
        actions: jnp.ndarray,
        is_training: bool = True,
    ):
        """FDM takes the prev states and latent action and predicts the next state

        T is the length of the context provided. For LAPO, T=2, just o_t-1 and o_t

        Input:
            prev_states: (B, T, D) or (B, T, C, H, W) for image inputs
            actions: (B, D_L)

        Output:
            next_state_pred: (B, T, D) or (B, T, C, H, W)
        """
        logging.info("inside LatentFDM")
        prev_states = einops.rearrange(prev_states, "b t h w c -> b (t c) h w")
        h, w = prev_states.shape[2], prev_states.shape[3]
        actions = einops.rearrange(actions, "b dl -> b dl 1 1")
        action_expand = einops.repeat(actions, "b dl 1 1 -> b dl h w", h=h, w=w)
        x = jnp.concatenate([prev_states, action_expand], axis=1)
        logging.info(f"shape after concat: {x.shape}")

        _, intermediates = self.state_embed(
            x, is_training=is_training, return_intermediate=True
        )
        embeddings = intermediates[-1]

        # inject actions into the middle of the u-net

        intermediates[-1] = actions
        intermediates[-1] = einops.repeat(
            intermediates[-1],
            "b t 1 1 -> b t h w",
            h=embeddings.shape[-1],
            w=embeddings.shape[-1],
        )
        next_state_pred = self.state_decoder(
            embeddings, intermediates=intermediates, is_training=is_training
        )
        # out = nn.tanh(out) / 2
        return next_state_pred


@dataclasses.dataclass
class LatentActionIDM(hk.Module):
    def __init__(
        self,
        config: ConfigDict,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs,
    ):
        """Inverse Dynamics Model in LAPO with Quantization

        Args:
          pass
        """
        super().__init__(name="LatentActionIDM")

        init_kwargs = dict(w_init=w_init, b_init=b_init)
        self.image_obs = config.image_obs

        if self.image_obs:
            self.state_embed = ImpalaCNN(**config.image_encoder_config, **init_kwargs)
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

        self.policy_head = hk.nets.MLP(
            list(config.layer_sizes) + [config.code_dim],
            activation=nn.gelu,
            name="policy_head",
            activate_final=True,
            **init_kwargs,
        )

    def __call__(self, states: jnp.ndarray, is_training: bool = True):
        """IDM takes the state and next state and predicts the action
        IDM predicts the latent action (z_t) given o_t-1, o_t and o_t+1

        Use IMPALA CNN for encoding the images

        T=3 for LAPO, o_t-1, o_t, o_t+1

        Input:
            states: (B, T, D) or (B, T, H, W, C)

        Output:
            vq_outputs: dict
        """
        if self.image_obs:
            # compute T and C dimension
            # first transpose
            states = einops.rearrange(states, "b t h w c -> b (t c) h w")
            # run it through the ImpalaCNN encoder
            # [B, (T*C), H, W] -> [B, D]
            state_embeds = self.state_embed(states, is_training=is_training)
        else:
            state_embeds = self.state_embed(states)

        # predict latent actions
        latent_actions = self.policy_head(nn.gelu(state_embeds))

        # compute quantized latent actions
        vq_outputs = self.vq(latent_actions, is_training)
        return vq_outputs
