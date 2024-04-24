import dataclasses
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
from varibad_jax.models.common import ImageEncoder
from ml_collections.config_dict import ConfigDict

@dataclasses.dataclass
class RND(hk.Module):
    def __init__(
        self,
        config: ConfigDict,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs
    ):
        """Random Network Distillation (RND) module

        Args:
          pass
        """
        super().__init__(name="RND")

        init_kwargs = dict(w_init=w_init, b_init=b_init)

        self.image_obs = config.image_obs
        self.embedding_dim = config.embedding_dim
        self.rnd_output_dim = config.rnd_output_dim 
        self.layer_sizes = config.layer_sizes

        self.latent_embed = hk.Linear(
            self.embedding_dim, name="latent_embed", **init_kwargs
        )

        if self.image_obs:
            self.state_embed = ImageEncoder(**image_encoder_config, **init_kwargs)
        else:
            self.state_embed = hk.Linear(
                self.embedding_dim, name="state_embed", **init_kwargs
            )

        self.output_mlp = hk.nets.MLP(
            list(self.layer_sizes) + [self.rnd_output_dim],
            activation=nn.gelu,
            name="rnd_output_mlp",
            activate_final=False,
            **init_kwargs,
        )

    def __call__(
        self,
        states: jnp.ndarray,
        latents: jnp.ndarray = None,
        is_training: bool = True,
    ):
        """Forwards the hyper-state (state, latent) separately and concatenates them"""
        state_embed = self.state_embed(states)

        rnd_input = state_embed

        if latents is not None:
            latent_embed = self.latent_embed(latents)

            rnd_input = jnp.concatenate([rnd_input, latent_embed], axis=-1)

        rnd_input = nn.gelu(rnd_input)

        rnd_output = self.output_mlp(rnd_input)
        return rnd_input
