from absl import logging
from typing import List, Optional
import einops
import haiku as hk
import flax.linen as nn
import jax.numpy as jnp


class DownsamplingBlock(hk.Module):
    def __init__(
        self,
        num_channels: int,
        add_bn: bool = False,
        add_residual: bool = False,
        add_max_pool: bool = False,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str = "SAME",
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs,
    ):
        super().__init__()
        self.add_bn = add_bn
        self.add_residual = add_residual
        self.add_max_pool = add_max_pool
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_kwargs = dict(w_init=w_init, b_init=b_init)

    def __call__(self, x, is_training=True):
        # logging.info("inside downsampling block")
        x = hk.Conv2D(
            output_channels=self.num_channels,
            kernel_shape=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            data_format="NCHW",
            **self.init_kwargs,
        )(x)
        # logging.info(f"conv shape: {x.shape}")
        if self.add_bn:
            x = hk.BatchNorm(create_offset=True, create_scale=True, decay_rate=0.9)(
                x, is_training
            )
        if self.add_residual:
            x = ResidualBlock(self.num_channels, **self.init_kwargs)(x)
            # logging.info(f"residual shape: {x.shape}")
        if self.add_max_pool:
            x = hk.MaxPool(
                window_shape=(2, 2),
                strides=(2, 2),
                padding="SAME",
            )(x)
            # logging.info(f"pool shape: {x.shape}")
        x = nn.gelu(x)
        return x


class UpsamplingBlock(hk.Module):
    def __init__(
        self,
        num_channels: int,
        add_bn: bool = False,
        add_residual: bool = False,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str = "SAME",
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs,
    ):
        super().__init__()
        self.add_bn = add_bn
        self.add_residual = add_residual
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_kwargs = dict(w_init=w_init, b_init=b_init)

    def __call__(self, x, is_training=True):
        x = hk.Conv2DTranspose(
            output_channels=self.num_channels,
            kernel_shape=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            data_format="NCHW",
            **self.init_kwargs,
        )(x)
        if self.add_bn:
            x = hk.BatchNorm(create_offset=True, create_scale=True, decay_rate=0.9)(
                x, is_training
            )
        if self.add_residual:
            x = ResidualBlock(self.num_channels, **self.init_kwargs)(x)
        x = nn.gelu(x)
        return x


class ResidualBlock(hk.Module):
    """Residual block."""

    def __init__(self, num_channels, w_init, b_init, **kwargs):
        super().__init__()
        self._num_channels = num_channels
        self.init_kwargs = dict(w_init=w_init, b_init=b_init)

    def __call__(self, x):
        main_branch = hk.Sequential(
            [
                nn.gelu,
                hk.Conv2D(
                    self._num_channels // 2,
                    kernel_shape=[3, 3],
                    stride=[1, 1],
                    padding="SAME",
                    data_format="NCHW",
                    **self.init_kwargs,
                ),
                nn.gelu,
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


class ImageEncoder(hk.Module):
    def __init__(
        self,
        embedding_dim: int,
        arch: List[List[int]] = [],
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        scale: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.arch = arch
        self.scale = scale
        self.init_kwargs = dict(w_init=w_init, b_init=b_init)
        self.kwargs = kwargs

    def __call__(self, x, is_training=True, return_intermediate=False):
        logging.info(f"encoder is training: {is_training}")
        reshape = x.ndim > 4
        if reshape:
            # last three are CHW
            lead_dims = x.shape[:-3]
            x = einops.rearrange(x, "... C H W -> (...) C H W")

        intermediate = []
        for i, spec in enumerate(self.arch):
            x = DownsamplingBlock(
                num_channels=spec[0] * self.scale,
                kernel_size=spec[1],
                stride=spec[2],
                padding=spec[3],
                **self.kwargs,
            )(x, is_training)
            intermediate.append(x)
            logging.info(f"encoder layer {i} shape: {x.shape}")

        # flatten and reembed
        x = hk.Flatten(preserve_dims=1)(x)
        x = hk.Linear(self.embedding_dim, **self.init_kwargs)(x)

        logging.info(f"encoder embedding shape: {x.shape}")
        if reshape:
            # restore leading dimensions, use lead_dims
            x = x.reshape(lead_dims + (x.shape[-1],))

        if not return_intermediate:
            return x

        return x, intermediate


class ImageDecoder(hk.Module):
    def __init__(
        self,
        arch: List[List[int]] = [],
        add_residual: bool = False,
        add_bn: bool = False,
        num_output_channels: int = 3,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs,
    ):
        super().__init__()
        self.arch = arch
        self.add_bn = add_bn
        self.add_residual = add_residual
        self.num_output_channels = num_output_channels
        self.init_kwargs = dict(w_init=w_init, b_init=b_init)

    def __call__(self, x, intermediates: List[jnp.ndarray] = None, is_training=True):
        logging.info(f"decoder is training: {is_training}")

        if x.ndim > 4:
            # last three are CHW
            lead_dims = x.shape[:-3]
            x = einops.rearrange(x, "... C H W -> (...) C H W")

        for i in range(len(self.arch)):
            if intermediates is not None:
                x = jnp.concatenate([x, intermediates[-i - 1]], axis=1)
            x = hk.Conv2DTranspose(
                output_channels=self.arch[i][0],
                kernel_shape=self.arch[i][1],
                stride=self.arch[i][2],
                padding=self.arch[i][3],
                data_format="NCHW",
                **self.init_kwargs,
            )(x)
            logging.info(f"decoder layer {i} shape: {x.shape}")
            if self.add_bn:
                x = hk.BatchNorm(create_offset=True, create_scale=True, decay_rate=0.9)(
                    x, is_training
                )
            x = nn.gelu(x)

        # last layer
        # this is just to get the right number of channels, doesn't change the spatial dimensions
        # also remember no activation after this
        x = hk.Conv2D(
            output_channels=self.num_output_channels,
            kernel_shape=1,
            stride=1,
            padding="SAME",
            data_format="NCHW",
            **self.init_kwargs,
        )(x)
        logging.info(f"final conv layer shape: {x.shape}")

        if x.ndim > 4:
            # restore leading dimensions, use lead_dims
            x = x.reshape(lead_dims + x.shape[1:])

        return x
