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

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional
import flax.linen as nn
import chex


@chex.dataclass
class LAMOutputs:
    z_e: jnp.ndarray
    z_q: jnp.ndarray
    obs_pred: jnp.ndarray
    codebook_loss: jnp.ndarray


def latent_action_model(config, obs):
    """
    Args:
        config: dict
        obs: jnp.ndarray, shape [B, T, H, W, C]
    """
    model = LatentActionModel(config)
    # we will encode the full obs seq x_1 ... x_t, x_t+1
    z_e = model.encode(obs)
    quantize_output = model.quantize(z_e)
    z_q = quantize_output["quantize"]  # maybe this is our latent action
    # are the latent actions the encoding indices from the quantizer?
    # latent_actions = quantize_output["encoding_indices"]
    codebook_loss = quantize_output["codebook_loss"]
    obs_pred = model.decode(z_q)
    return LAMOutputs(
        z_e=z_e,
        z_q=z_q,
        obs_pred=obs_pred,
        codebook_loss=codebook_loss,
    )


class LatentActionModel(hk.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.quantizer = QuantizedCodebook(
            num_codes=config["num_codes"],
            code_dim=config["code_dim"],
            commitment_loss=config["commitment_loss"],
        )
        self.decoder = Decoder(config)

    def encode(self, obs):
        return self.encoder(obs)

    def quantize(self, z_e):
        return self.quantizer(z_e)["quantize"]

    def decode(self, z_q):
        return self.decoder(z_q)


class QuantizedCodebook(hk.Module):
    def __init__(
        self,
        embed_size_K: int,
        embed_dim_D: int,
        commitment_loss: float,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.K = embed_size_K
        self.D = embed_dim_D
        self.beta = commitment_loss

        initializer = hk.initializers.VarianceScaling(distribution="uniform")
        self.codebook = hk.get_parameter("codebook", (self.K, self.D), init=initializer)

    def __call__(self, inputs) -> dict[str, jnp.ndarray]:
        """ """
        # input shape A1 x ... x An x D
        # shape N x D, N = A1 * ... * An
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
            "quantize": quantize,
            "encoding_indices": encoding_indices,
        }

    def embed(self, indices):
        outshape = indices.shape + (self.D,)
        x = self.codebook[indices].reshape(outshape)
        return x
