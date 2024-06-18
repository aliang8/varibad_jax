import chex
import jax
from absl import logging
import flax.linen as nn
import haiku as hk
import jax.numpy as jnp
from typing import List
import einops


class ResidualBlock(hk.Module):
    """Residual block."""

    def __init__(self, num_channels, w_init, b_init, **kwargs):
        super().__init__()
        self._num_channels = num_channels
        self.init_kwargs = dict(w_init=w_init, b_init=b_init)

    def __call__(self, x):
        main_branch = hk.Sequential(
            [
                nn.relu,
                hk.Conv2D(
                    self._num_channels // 2,
                    kernel_shape=[3, 3],
                    stride=[1, 1],
                    padding="SAME",
                    data_format="NCHW",
                    **self.init_kwargs,
                ),
                nn.relu,
                hk.Conv2D(
                    self._num_channels,
                    kernel_shape=[3, 3],
                    stride=[1, 1],
                    padding="SAME",
                    data_format="NCHW",
                    **self.init_kwargs,
                ),
            ]
        )
        return main_branch(x) + x


class ImpalaCNN(hk.Module):
    def __init__(
        self,
        out_channels: List[int],
        out_features: int,
        scale: int = 1,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs,
    ):
        super().__init__()
        self._out_channels = out_channels
        self.out_features = out_features
        self.scale = scale
        self.init_kwargs = dict(w_init=w_init, b_init=b_init)

    def __call__(self, x, is_training: bool = False):
        logging.info("inside ImpalaCNN")
        logging.info(f"input shape: {x.shape}")
        for indx, out_ch in enumerate(self._out_channels):
            x = hk.Conv2D(
                out_ch * self.scale,
                kernel_shape=[3, 3],
                stride=[1, 1],
                padding="VALID",
                data_format="NCHW",
                **self.init_kwargs,
            )(x)
            logging.info(f"shape after conv {indx}: {x.shape}")
            x = hk.MaxPool(window_shape=[3, 3], strides=[2, 2], padding="VALID")(x)
            logging.info(f"shape after pool {indx}: {x.shape}")

            x = ResidualBlock(out_ch * self.scale, **self.init_kwargs)(x)
            x = ResidualBlock(out_ch * self.scale, **self.init_kwargs)(x)
            logging.info(f"shape after conv seq {indx}: {x.shape}")

        x = hk.Flatten()(x)
        x = nn.relu(x)
        x = hk.Linear(self.out_features, **self.init_kwargs)(x)
        return x


@chex.dataclass
class VQOutput:
    quantize: jnp.ndarray
    loss: jnp.ndarray
    perplexity: jnp.ndarray
    encoding_indices: jnp.ndarray
    encodings: jnp.ndarray
    latent_actions: jnp.ndarray = None


class VQEmbeddingEMA(hk.Module):
    def __init__(
        self,
        epsilon: int = 1e-5,
        num_codebooks: int = 2,
        num_embs: int = 64,
        emb_dim: int = 16,
        num_discrete_latents: int = 4,
        decay: float = 0.999,
        commitment_loss: float = 0.05,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_codebooks = num_codebooks
        self.num_embs = num_embs
        self.emb_dim = emb_dim
        self.num_discrete_latents = num_discrete_latents
        self.decay = decay
        self.commitment_loss = commitment_loss

    def forward_2d(self, x, is_training=True):
        embedding = hk.get_state(
            "embedding",
            shape=(self.num_codebooks, self.num_embs, self.emb_dim),
            dtype=jnp.float32,
            init=hk.initializers.RandomUniform(
                -1.0 / self.num_embs * 5, 1.0 / self.num_embs * 5
            ),
        )
        ema_count = hk.get_state(
            "ema_count",
            shape=(self.num_codebooks, self.num_embs),
            dtype=jnp.int32,
            init=jnp.zeros,
        )

        ema_weight = hk.get_state(
            "ema_weight",
            shape=(self.num_codebooks, self.num_embs, self.emb_dim),
            dtype=jnp.float32,
            init=hk.initializers.RandomUniform(
                -1.0 / self.num_embs * 5, 1.0 / self.num_embs * 5
            ),
        )

        B, C, H, W = x.shape
        N, M, D = embedding.shape
        assert C == N * D

        x = einops.rearrange(x, "b (n d) h w -> n b h w d", n=N, d=D)

        # x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        # x_flat = x.detach().reshape(N, -1, D)

        x_flat = einops.rearrange(x, "n b h w d -> n (b h w) d")
        x_flat = jax.lax.stop_gradient(x_flat)

        to_add = jnp.expand_dims(jnp.sum(embedding**2, axis=2), axis=1)
        to_add += jnp.sum(x_flat**2, axis=2, keepdims=True)

        embedding_t = jnp.swapaxes(embedding, 1, 2)
        b1b2 = jax.lax.batch_matmul(x_flat, embedding_t)
        alpha = -2.0
        beta = 1.0
        distances = beta * to_add + alpha * b1b2

        indices = jnp.argmin(distances, axis=-1)
        encodings = jax.nn.one_hot(indices, M).astype(jnp.float32)
        indices_exp = einops.repeat(indices, "n b -> n b d", d=D)
        quantized = jnp.take_along_axis(embedding, indices_exp, axis=1)
        quantized = quantized.reshape(x.shape)

        if is_training:
            new_ema_count = self.decay * ema_count + (1 - self.decay) * jnp.sum(
                encodings, axis=1
            )
            n = jnp.sum(new_ema_count, axis=-1, keepdims=True)
            new_ema_count = (new_ema_count + self.epsilon) / (n + M * self.epsilon) * n
            encodings_t = jnp.swapaxes(encodings, 1, 2)
            dw = jax.lax.batch_matmul(encodings_t, x_flat)
            new_ema_weight = self.decay * ema_weight + (1 - self.decay) * dw

            new_embeddings = new_ema_weight / jnp.expand_dims(new_ema_count, axis=-1)

            hk.set_state("ema_count", new_ema_count)
            hk.set_state("ema_weight", new_ema_weight)
            hk.set_state("embedding", new_embeddings)

        e_latent_loss = jnp.mean((x - jax.lax.stop_gradient(quantized)) ** 2)
        loss = self.commitment_loss * e_latent_loss
        quantized = jax.lax.stop_gradient(quantized) + (x - jax.lax.stop_gradient(x))
        avg_probs = jnp.mean(encodings, axis=1)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10), axis=-1))

        quantized = einops.rearrange(quantized, "n b h w d -> b (n d) h w", n=N, d=D)

        indices = einops.rearrange(indices, "n (b h w) -> b n h w", n=N, b=B, h=H, w=W)

        return quantized, loss, perplexity.sum(), indices

    def __call__(self, x, is_training=True):
        bs = x.shape[0]
        x = einops.rearrange(
            x,
            "b (n d l) -> b (n d) l 1",
            b=bs,
            n=self.num_codebooks,
            d=self.emb_dim,
            l=self.num_discrete_latents,
        )
        z_q, loss, perplexity, indices = self.forward_2d(x, is_training=is_training)
        z_q = z_q.reshape(
            bs,
            self.num_codebooks * self.num_discrete_latents * self.emb_dim,
        )

        encodings = self.inds_to_z_q(indices)

        return VQOutput(
            quantize=z_q,
            loss=loss,
            perplexity=perplexity,
            encoding_indices=indices,
            encodings=encodings,
        )

    def inds_to_z_q(self, indices):
        """look up quantization inds in embedding"""
        embedding = hk.get_state(
            "embedding",
            shape=(self.num_codebooks, self.num_embs, self.emb_dim),
            dtype=jnp.float32,
            init=hk.initializers.RandomNormal(
                -1.0 / self.num_embs * 5, 1.0 / self.num_embs * 5
            ),
        )

        N, M, D = embedding.shape
        B, N_, H, W = indices.shape
        assert N == N_

        # N ... num_codebooks
        # M ... num_embs
        # D ... emb_dim
        # H ... num_discrete_latents (kinda)
        inds_flat = einops.rearrange(indices, "b n h w -> n (b h w)")
        inds_flat = einops.repeat(inds_flat, "n b -> n b d", d=D)
        quantized = jnp.take_along_axis(embedding, inds_flat, axis=1)
        quantized = quantized.reshape(N, B, H, W, D)
        quantized = einops.rearrange(quantized, "n b h w d -> b (n d) h w", n=N, d=D)

        return (
            quantized  # shape is (B, num_codebooks * emb_dim, num_discrete_latents, 1)
        )
