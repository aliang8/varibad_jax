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
        kernel_shapes: tuple,
        padding: str = "VALID",
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
    ):
        super().__init__()
        net = []
        for out_channels in output_channels:
            net.extend(
                [
                    hk.Conv2D(
                        out_channels,
                        kernel_shapes,
                        padding=padding,
                        w_init=w_init,
                        b_init=b_init,
                        data_format="NHWC",
                    ),
                    nn.gelu,
                ]
            )

        self.encoder = hk.Sequential(
            [
                *net,
                hk.Flatten(preserve_dims=1),
                hk.Linear(embedding_dim, w_init=w_init, b_init=b_init),
            ]
        )

    def __call__(self, x):
        if x.ndim >= 4:
            # last three are H, W, C
            lead_dims = x.shape[:-3]
            enc_input = einops.rearrange(x, "... H W C -> (...) H W C")
        else:
            enc_input = x

        embedding = self.encoder(enc_input)

        if x.ndim >= 4:
            # restore leading dimensions, use lead_dims
            embedding = embedding.reshape(lead_dims + (-1,))

        return embedding


class ResBlock(hk.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

    def __call__(self, x, is_training: bool) -> jnp.ndarray:
        res = hk.Conv2D(self.dim, self.kernel_size)(x)
        res = hk.BatchNorm(True, True, 0.9)(res, is_training)
        res = nn.relu(res)
        res = hk.Conv2D(self.dim, self.kernel_size)(res)
        res = hk.BatchNorm(True, True, 0.9)(res, is_training)
        x += res
        x = nn.relu(x)
        return x


class CnnEncoder(hk.Module):
    def __init__(
        self,
        out_channels: int,
        downscale_level: int,
        res_layers: int = 1,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.downscale_level = downscale_level
        self.res_layers = res_layers
        self.kernel_size = kernel_size

    def __call__(self, x, is_training: bool) -> jnp.ndarray:
        if x.ndim >= 4:
            # last three are H, W, C
            lead_dims = x.shape[:-3]
            x = einops.rearrange(x, "... H W C -> (...) H W C")
        else:
            lead_dims = None
            x = x

        for i in range(self.downscale_level - 1, -1, -1):
            num_channels = self.out_channels // (2**i)
            x = hk.Conv2D(num_channels, self.kernel_size, stride=2)(x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
            x = nn.relu(x)
            for _ in range(self.res_layers):
                x = ResBlock(num_channels, self.kernel_size)(x, is_training)

        if lead_dims is not None:
            # restore leading dimensions, use lead_dims
            x = x.reshape(lead_dims + (-1,))

        return x


class CnnDecoder(hk.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_level: int,
        res_layers: int = 1,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale_level = upscale_level
        self.res_layers = res_layers
        self.kernel_size = kernel_size

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        if x.ndim >= 4:
            # last three are H, W, C
            lead_dims = x.shape[:-3]
            x = einops.rearrange(x, "... H W C -> (...) H W C")
        else:
            lead_dims = None
            x = x

        for i in range(self.upscale_level - 1):
            num_channels = self.in_channels // (2**i)
            x = hk.Conv2DTranspose(num_channels, self.kernel_size, stride=2)(x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training)
            x = nn.relu(x)
            for _ in range(self.res_layers):
                x = ResBlock(num_channels, self.kernel_size)(x, is_training)

        # import ipdb

        # ipdb.set_trace()

        # [4, 4, 32]

        # [5, 5, 2]
        x = hk.Conv2DTranspose(self.out_channels, 2, stride=1, padding="VALID")(x)

        # # -> [8, 8, 2]
        # x = hk.Conv2DTranspose(self.out_channels, self.kernel_size, stride=2)(x)

        # # -> [7, 7, 2]
        # x = hk.Conv2D(self.out_channels, 2, stride=1, padding="VALID")(x)

        if lead_dims is not None:
            # restore leading dimensions, use lead_dims
            x = x.reshape(lead_dims + x.shape[-3:])

        return x
