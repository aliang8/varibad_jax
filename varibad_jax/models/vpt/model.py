import dataclasses
from typing import Any, Callable, List, NamedTuple, Optional, Tuple
from absl import logging
import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
import einops
from ml_collections.config_dict import ConfigDict

from varibad_jax.models.common import ImageEncoder
from varibad_jax.models.transformer_encoder import TransformerEncoder
from varibad_jax.agents.common import ActionHead


@dataclasses.dataclass
class IDM(hk.Module):
    def __init__(
        self,
        config: ConfigDict,
        is_continuous: bool,
        action_dim: int,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs,
    ):
        """Inverse Dynamics Model for VPT

        Args:
            pass
        """
        super().__init__(name="IDM")

        init_kwargs = dict(w_init=w_init, b_init=b_init)
        self.image_obs = config.image_obs
        self.use_transformer = config.use_transformer

        if self.use_transformer:
            self.transformer = TransformerEncoder(
                **config.transformer_config, **init_kwargs
            )
            if self.image_obs:
                # Used to encode the image before feeding as input to the TransformerEncoder
                self.state_embed = ImageEncoder(
                    **config.image_encoder_config, **init_kwargs
                )
            else:
                # Use single linear layer if we are using Transformer Encoder
                self.state_embed = hk.Linear(config.embedding_dim, **init_kwargs)
        else:
            if self.image_obs:
                self.state_embed = ImageEncoder(
                    **config.image_encoder_config, **init_kwargs
                )
                # self.state_embed = ImpalaCNN(
                #     **config.image_encoder_config, **init_kwargs
                # )
            else:
                # MLP
                self.state_embed = hk.nets.MLP(
                    list(config.state_embed_mlp_sizes) + [config.embedding_dim],
                    activation=nn.gelu,
                    **init_kwargs,
                    activate_final=False,
                )

        # Predict ground truth action
        self.action_head = ActionHead(
            gaussian_policy=config.gaussian_policy,
            is_continuous=is_continuous,
            action_dim=action_dim,
            **init_kwargs,
        )

    def __call__(self, states: jnp.ndarray, is_training: bool = True):
        """IDM takes the state and next state and predicts the action

        Input:
            states: (B, T, D) or (B, T, H, W, C) if state or image observations

        Output:
            action_output
        """
        if self.use_transformer:
            # first embed the states
            if self.image_obs:
                b = states.shape[0]
                states = einops.rearrange(states, "b t h w c -> (b t) c h w")
                state_embeds = self.state_embed(states, is_training=is_training)
                state_embeds = einops.rearrange(state_embeds, "(b t) d -> b t d", b=b)
            else:
                state_embeds = self.state_embed(states)

            state_embeds = nn.gelu(state_embeds)
            state_embeds = self.transformer(state_embeds, is_training=is_training)
            state_embeds = state_embeds[:, 1:]
            # ignore the first timestep embedding
            # the resulting VQ actions should have T-1 outputs because we are
            # predicting the action that took us from t to t+1
            logging.info(f"shape after transformer enc: {state_embeds.shape}")
        else:
            if self.image_obs:
                # combine T and C dimension so that will be channel input to the CNN
                states = einops.rearrange(states, "b t h w c -> b (t c) h w")
                # run it through the ImpalaCNN encoder
                state_embeds = self.state_embed(states, is_training=is_training)
            else:
                # [B, T, state_dim]
                # flatten the last two dimensions
                states = einops.rearrange(states, "b t d -> b (t d)")
                state_embeds = self.state_embed(states)

        # predict actions
        actions = self.action_head(nn.gelu(state_embeds))
        return actions
