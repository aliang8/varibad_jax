import dataclasses
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
from varibad_jax.models.common import ImageEncoder
from ml_collections.config_dict import ConfigDict

@dataclasses.dataclass
class LatentFDM(hk.Module):
    def __init__(
        self,
        config: ConfigDict,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs
    ):
        """Forward Dynamics Model in LAPO, they call this a World Model

        Args:
          pass
        """
        super().__init__(name="LatentFDM")

        init_kwargs = dict(w_init=w_init, b_init=b_init)

        self.image_obs = config.image_obs
        self.embedding_dim = config.embedding_dim
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
        actions: jnp.ndarray,
        is_training: bool = True,
    ):
        """FDM takes the state and latent action and predicts the next state
        
        states: (B, T, D) or (B, T, C, H, W)
        actions: (B, T)
        """
        state_embed = self.state_embed(states)


        return rnd_input


@dataclasses.dataclass
class LatentActionIDM(hk.Module):
    def __init__(
        self,
        config: ConfigDict,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs
    ):
        """Inverse Dynamics Model in LAPO with Quantization

        Args:
          pass
        """
        super().__init__(name="LatentActionIDM")

        init_kwargs = dict(w_init=w_init, b_init=b_init)

        self.image_obs = config.image_obs
        self.embedding_dim = config.embedding_dim
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
            
        # see VQVAE in https://arxiv.org/abs/1711.00937
        # the EMA version uses an exponential moving average of the embedding vectors
        self.vq = hk.nets.VectorQuantizerEMA(
            embedding_dim=self.embedding_dim,
            num_embeddings=config.num_embeddings,
            commitment_cost=config.commitment_cost,
            decay=config.ema_decay,
            name="VQEMA"
        )

        self.output_mlp = hk.nets.MLP(
            list(self.layer_sizes) + [self.rnd_output_dim],
            activation=nn.gelu,
            name="rnd_output_mlp",
            activate_final=False,
            **init_kwargs,
        )
        
    def __call__(self, states: jnp.ndarray, next_states: jnp.ndarray, is_training: bool = True):
        """IDM takes the state and next state and predicts the action
        
        states: (B, T, D) or (B, T, C, H, W)
        next_states: (B, T, D) or (B, T, C, H, W)
        """
        state_embed = self.state_embed(states)
        next_state_embed = self.state_embed(next_states)
        
        policy_input = jnp.concatenate([state_embed, next_state_embed], axis=-1)
        policy_input = nn.gelu(policy_input)
        
        # predict latent actions
        latent_actions = self.policy_head(policy_input)
        
        # compute quantized latent actions
        quantize, loss, perplexity, encodings, encoding_indices = self.vq(latent_actions, is_training)
        
        model_outputs = LAMOutputs(
            z_e=encodings, 
            z_q=quantize,
            obs_pred=None,
            codebook_loss=loss,
            encoding_indices=encoding_indices
        )
        
        return model_outputs

        
