from absl import logging
from typing import List, Optional
import einops
import haiku as hk
import flax.linen as nn
import jax.numpy as jnp


class ImageEncoder(hk.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_channels: List[int],
        kernel_shapes: List[int],
        strides: List[int],
        padding: List[str] = [],
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_channels = output_channels
        self.kernel_shapes = kernel_shapes
        self.strides = strides
        self.padding = padding
        self.init_kwargs = dict(w_init=w_init, b_init=b_init)

    def __call__(self, x, is_training=True, return_intermediate=False):
        logging.info(f"encoder is training: {is_training}")
        if x.ndim > 4:
            # last three are CHW
            lead_dims = x.shape[:-3]
            x = einops.rearrange(x, "... C H W -> (...) C H W")

        intermediate = []
        for i in range(len(self.output_channels)):
            x = hk.Conv2D(
                self.output_channels[i],
                kernel_shape=self.kernel_shapes[i],
                stride=self.strides[i],
                padding=self.padding[i],
                data_format="NCHW",
                **self.init_kwargs,
            )(x)
            logging.info(f"encoder layer {i} shape: {x.shape}")
            # x = hk.BatchNorm(create_offset=True, create_scale=True, decay_rate=0.9)(x, is_training)
            x = nn.gelu(x)
            intermediate.append(x)

        # flatten and reembed
        embedding = hk.Flatten(preserve_dims=1)(x)
        embedding = hk.Linear(self.embedding_dim, **self.init_kwargs)(embedding)

        logging.info(f"encoder embedding shape: {embedding.shape}")
        if x.ndim > 4:
            # restore leading dimensions, use lead_dims
            embedding = embedding.reshape(lead_dims + (embedding.shape[-1],))

        if not return_intermediate:
            return embedding

        return embedding, intermediate


class ImageDecoder(hk.Module):
    def __init__(
        self,
        output_channels: List[int],
        kernel_shapes: List[int],
        strides: List[int],
        padding: str = "VALID",
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs,
    ):
        super().__init__()
        self.output_channels = output_channels
        self.kernel_shapes = kernel_shapes
        self.strides = strides
        self.padding = padding
        self.init_kwargs = dict(w_init=w_init, b_init=b_init)

    def __call__(self, x, intermediates: List[jnp.ndarray], is_training=True):
        logging.info(f"decoder is training: {is_training}")

        if x.ndim > 4:
            # last three are CHW
            lead_dims = x.shape[:-3]
            x = einops.rearrange(x, "... C H W -> (...) C H W")

        for i in range(len(self.output_channels)):
            x = jnp.concatenate([x, intermediates[-i - 1]], axis=1)
            x = hk.Conv2DTranspose(
                self.output_channels[i],
                self.kernel_shapes[i],
                stride=self.strides[i],
                padding=self.padding[i],
                data_format="NCHW",
                **self.init_kwargs,
            )(x)
            logging.info(f"decoder layer {i} shape: {x.shape}")
            # x = hk.BatchNorm(create_offset=True, create_scale=True, decay_rate=0.9)(x, is_training)
            x = nn.gelu(x)

        if x.ndim > 4:
            # restore leading dimensions, use lead_dims
            x = x.reshape(lead_dims + x.shape[1:])

        return x
