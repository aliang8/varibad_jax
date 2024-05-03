from absl import logging
import flax.linen as nn
import haiku as hk
from typing import List


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
        impala_scale: int = 1,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs,
    ):
        super().__init__()
        self._out_channels = out_channels
        self.out_features = out_features
        self.impala_scale = impala_scale
        self.init_kwargs = dict(w_init=w_init, b_init=b_init)

    def __call__(self, x, is_training: bool = False):
        logging.info("inside ImpalaCNN")
        logging.info(f"input shape: {x.shape}")
        for indx, out_ch in enumerate(self._out_channels):
            x = hk.Conv2D(
                out_ch * self.impala_scale,
                kernel_shape=[3, 3],
                stride=[1, 1],
                padding="VALID",
                data_format="NCHW",
                **self.init_kwargs,
            )(x)
            logging.info(f"shape after conv {indx}: {x.shape}")
            x = hk.MaxPool(window_shape=[3, 3], strides=[2, 2], padding="VALID")(x)
            logging.info(f"shape after pool {indx}: {x.shape}")

            x = ResidualBlock(out_ch * self.impala_scale, **self.init_kwargs)(x)
            x = ResidualBlock(out_ch * self.impala_scale, **self.init_kwargs)(x)
            logging.info(f"shape after conv seq {indx}: {x.shape}")

        x = hk.Flatten()(x)
        x = nn.relu(x)
        x = hk.Linear(self.out_features, **self.init_kwargs)(x)
        return x
