import einops
import haiku as hk
import flax.linen as nn


class ImageEncoder(hk.Module):
    def __init__(
        self,
        embedding_dim: int,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
    ):
        super().__init__()
        output_channels = [16, 32, 64]
        kernel_shape = (2, 2)
        padding = "VALID"

        self.encoder = hk.Sequential(
            [
                *[
                    hk.Conv2D(
                        output_channels[i],
                        kernel_shape,
                        padding=padding,
                        w_init=w_init,
                    )
                    for i in range(len(output_channels))
                ],
                nn.gelu,
                hk.Flatten(preserve_dims=1),
                hk.Linear(embedding_dim, w_init=w_init, b_init=b_init),
            ]
        )

    def __call__(self, x):
        if x.ndim >= 5:
            # last three are C, H, W
            lead_dims = x.shape[:-3]
            enc_input = einops.rearrange(x, "... C H W -> (...) C H W")
        else:
            enc_input = x

        embedding = self.encoder(enc_input)

        if x.ndim >= 5:
            # restore leading dimensions, use lead_dims
            embedding = embedding.reshape(lead_dims + (-1,))

        return embedding
