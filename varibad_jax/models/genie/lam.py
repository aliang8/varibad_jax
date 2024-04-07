"""
Latent Action Model
https://arxiv.org/pdf/2402.15391v1.pdf

Encoder takes in input all previous frames and next frame: (x1, ..., xt, xt+1)
Outputs the latent actions: a1, ..., at
Decoder takes all previous frames and latent actions predicts next frame xt+1

We use a VQ-VAE objective so the latent is a small set of discrete codes. 

Encoder maps x -> z_e 
Quantizer maps z_e -> z_q
Decoder maps z_q -> x
"""

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Dict
import flax.linen as nn
import chex
import einops
from ml_collections.config_dict import ConfigDict
from varibad_jax.models.transformer_encoder import TransformerEncoder
from varibad_jax.models.common import ImageEncoder, ImageDecoder


@chex.dataclass
class LAMOutputs:
    z_e: jnp.ndarray
    z_q: jnp.ndarray
    obs_pred: jnp.ndarray
    codebook_loss: jnp.ndarray
    encoding_indices: jnp.ndarray


class LatentActionModel(hk.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        # encodes observations and outputs latent embedding of image
        self.image_encoder = ImageEncoder(**config.image_encoder_config)
        # encodes sequence of observations
        # (x1, ..., xT) -> (z_1, ..., z_T)
        self.encoder = TransformerEncoder(**config.transformer_config)
        self.quantizer = QuantizedCodebook(
            num_codes=config.num_codes,
            code_dim=config.code_dim,
            beta=config.beta,
        )
        self.mid_layer = hk.Linear(
            config.code_dim,
            w_init=hk.initializers.VarianceScaling(2.0),
            b_init=hk.initializers.Constant(0.0),
        )
        # decoder takes in both the output from codebook and the latent embedding
        # use convtranspose to reconstruct image
        self.decoder = ImageDecoder(**config.image_decoder_config)

    def __call__(self, obs, is_training=True):
        # obs - [B, T, H, W, C]

        # first encode the observations
        # we will encode the full obs seq x_1 ... x_t, x_t+1
        img_embedding = self.image_encoder(obs, is_training=is_training)
        logging.info(f"img_embedding shape: {img_embedding.shape}")
        z_e = self.encoder(img_embedding, deterministic=not is_training)
        logging.info(f"z_e shape: {z_e.shape}")

        # quantize the latent embedding
        quantize_output = self.quantizer(z_e)
        z_q = quantize_output["z_q"]
        logging.info(f"z_q shape: {z_q.shape}")

        # decode given z_e and z_q
        # TODO: maybe this is another transformer
        z_qe = jnp.concatenate([z_q, z_e], axis=-1)
        z_qe = self.mid_layer(z_qe)
        z_qe = nn.gelu(z_qe)
        logging.info(f"z_qe shape: {z_qe.shape}")
        # [B, T, D]

        # need to make this [B, T, 1, 1, D] for image decoding
        z_qe = einops.rearrange(z_qe, "B T D -> B T 1 1 D")

        # intermediate layer to encode z_q and z_e together
        obs_pred = self.decoder(z_qe, is_training=True)
        return LAMOutputs(z_e=z_e, obs_pred=obs_pred, **quantize_output)


class QuantizedCodebook(hk.Module):
    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        beta: float,
    ):
        super().__init__()
        self.K = num_codes
        self.D = code_dim
        self.beta = beta

        initializer = hk.initializers.VarianceScaling(distribution="uniform")
        self.codebook = hk.get_parameter("codebook", (self.K, self.D), init=initializer)

    def __call__(self, inputs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        # input shape A1 x ... x An x D
        # shape N x D, N = A1 * ... * An
        # flatten the first N dimensions except for the last
        flattened = jnp.reshape(inputs, (-1, self.D))

        # shape N x 1
        flattened_sqr = jnp.sum(flattened**2, axis=-1, keepdims=True)

        # shape 1 x K
        codeboook_sqr = jnp.sum(self.codebook**2, axis=-1, keepdims=True).T

        # shape N x K
        # distances = (a-b)^2 = a^2 - 2*a*b + b^2
        distances = flattened_sqr - 2 * (flattened @ self.codebook.T) + codeboook_sqr

        # shape A1 x ... x An
        encoding_indices = jnp.reshape(
            jnp.argmin(distances, axis=-1), inputs.shape[:-1]
        )

        # shape A1 x ... x An x D
        quantize = self.codebook[encoding_indices]

        # loss = ||sg[z_e(x)] - e|| + beta||z_e(x) - sg[e]||
        encoding_loss = jnp.mean((jax.lax.stop_gradient(inputs) - quantize) ** 2)
        commit_loss = jnp.mean((inputs - jax.lax.stop_gradient(quantize)) ** 2)
        loss = encoding_loss + self.beta * commit_loss

        # straight-through estimator for reconstruction loss
        quantize = inputs + jax.lax.stop_gradient(quantize - inputs)

        return {
            "codebook_loss": loss,
            "z_q": quantize,
            "encoding_indices": encoding_indices,
        }

    def embed(self, indices):
        outshape = indices.shape + (self.D,)
        x = self.codebook[indices].reshape(outshape)
        return x
