import dataclasses
from typing import Any, Callable, List, NamedTuple, Optional, Tuple
from absl import logging
import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
import einops
from ml_collections.config_dict import ConfigDict

from varibad_jax.models.common import ImageEncoder, ImageDecoder
from varibad_jax.models.lam.helpers import ImpalaCNN, VQEmbeddingEMA
from varibad_jax.models.transformer_encoder import (
    TransformerEncoder,
    TransformerDecoder,
)


@dataclasses.dataclass
class LatentFDM(hk.Module):
    def __init__(
        self,
        config: ConfigDict,
        state_dim: int,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs,
    ):
        """Forward Dynamics Model in LAPO, they call this a World Model in LAPO
        FDM predicts o_t+1 given o_t-k,..., o_t and z_t

        Use U-net style architecture following: https://github.com/schmidtdominik/LAPO/blob/main/lapo/models.py
        U-net the decoder input is concatenated with the encoder output at each level

        Args:
          pass
        """
        super().__init__(name="LatentFDM")

        init_kwargs = dict(w_init=w_init, b_init=b_init)
        self.image_obs = config.image_obs
        self.use_transformer = config.use_transformer

        if self.image_obs:
            # https://asiltureli.github.io/Convolution-Layer-Calculator/
            self.state_embed = ImageEncoder(
                **config.image_encoder_config, **init_kwargs
            )

            if self.use_transformer:
                self.fdm = TransformerDecoder(
                    **config.transformer_config, **init_kwargs
                )

            self.state_decoder = ImageDecoder(
                **config.image_decoder_config, **init_kwargs
            )
        else:
            self.embedding_dim = config.embedding_dim
            # self.state_embed = hk.nets.MLP(
            #     config.mlp_layer_sizes + [self.embedding_dim],
            #     **init_kwargs,
            #     activation=nn.gelu,
            # )
            # self.action_embed = hk.nets.MLP(
            #     config.mlp_layer_sizes + [self.embedding_dim],
            #     **init_kwargs,
            #     activation=nn.gelu,
            # )
            self.state_decoder = hk.nets.MLP(
                config.decoder_mlp_sizes + [state_dim],
                **init_kwargs,
                activation=nn.gelu,
                activate_final=False,
            )

    def __call__(
        self,
        context: jnp.ndarray,
        actions: jnp.ndarray,
        is_training: bool = True,
        **kwargs,
    ):
        """FDM takes the prev states and latent action and predicts the next state
        T is the length of the context provided.

        For LAPO, T=2, just o_t-1 and o_t
        If we use a Transformer, T is the full sequence or a larger context window

        Input:
            context: (B, T, D) or (B, T, C, H, W) for image inputs
            actions: (B, D_L)

        Output:
            next_state_pred: (B, T, D) or (B, T, C, H, W)
        """
        logging.info("inside LatentFDM")

        if self.image_obs:
            b, t, h, w = context.shape[:4]

            if self.use_transformer:
                # we will use a transformer decoder to predict the next state
                # and the latent actions as the value / key
                context = einops.rearrange(context, "b t h w c -> (b t) c h w")

                # first encode with the CNN backbone, u-net encoder
                embeddings, intermediates = self.state_embed(
                    context, is_training=is_training, return_intermediate=True
                )

                # reshape back to sequence
                embeddings = einops.rearrange(embeddings, "(b t) d -> b t d", b=b, t=t)

                # actions = einops.rearrange(actions, "b t dl -> (b t) dl 1 1")

                # src mask should be [B, T, L_1, L_2]
                src_mask = jnp.tril(jnp.ones((t, t)))
                src_mask = einops.repeat(src_mask, "l1 l2 -> b 1 l1 l2", b=b)
                tgt_mask = None

                import ipdb

                ipdb.set_trace()
                # run the input through the TransformerDecoder
                out = self.fdm(
                    embeddings,
                    actions,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask,
                    is_training=is_training,
                )

                # run through U-net decoder

                # reconstruct the next state with UNet decoder
                next_state_pred = einops.rearrange(
                    next_state_pred, "(b t) c h w -> b t c h w", b=b, t=t
                )

            else:
                context = einops.rearrange(context, "b t h w c -> b (t c) h w")
                actions = einops.rearrange(actions, "b dl -> b dl 1 1")

                action_expand = einops.repeat(actions, "b dl 1 1 -> b dl h w", h=h, w=w)

                x = jnp.concatenate([context, action_expand], axis=1)
                # x = context
                logging.info(f"shape after concat: {x.shape}")

                _, intermediates = self.state_embed(
                    x, is_training=is_training, return_intermediate=True
                )
                embeddings = intermediates[-1]
                # inject actions into the middle of the u-net (this is also done in LAPO)
                intermediates[-1] = actions

                next_state_pred = self.state_decoder(
                    embeddings,
                    intermediates=intermediates,
                    context=context,
                    is_training=is_training,
                )

        else:
            # flatten the last two dimensions
            context = einops.rearrange(context, "b t d -> b (t d)")
            # context_embed = self.state_embed(context)
            # action_embed = self.action_embed(actions)
            # model_input = jnp.concatenate([context_embed, action_embed], axis=-1)

            # model_input = actions
            model_input = jnp.concatenate([context, actions], axis=-1)

            # predict next state
            next_state_pred = self.state_decoder(model_input)

        return next_state_pred


@dataclasses.dataclass
class LatentActionIDM(hk.Module):
    def __init__(
        self,
        config: ConfigDict,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs,
    ):
        """Inverse Dynamics Model for Latent Action Model with VectorQuantization

        Args:
            pass
        """
        super().__init__(name="LatentActionIDM")

        init_kwargs = dict(w_init=w_init, b_init=b_init)
        self.config = config
        self.image_obs = config.image_obs
        self.use_transformer = config.use_transformer

        if self.use_transformer:
            self.transformer = TransformerEncoder(
                **config.transformer_config, **init_kwargs, casual=False
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
            else:
                # MLP
                self.state_embed = hk.nets.MLP(
                    list(config.state_embed_mlp_sizes) + [config.embedding_dim],
                    activation=nn.gelu,
                    **init_kwargs,
                    activate_final=False,
                )

        # Predict latent action before inputting to VQ
        self.latent_action_head = hk.nets.MLP(
            list(config.policy_mlp_sizes) + [config.code_dim],
            activation=nn.gelu,
            name="latent_action_head",
            activate_final=False,
            **init_kwargs,
        )

        # see VQVAE in https://arxiv.org/abs/1711.00937
        # the EMA version uses an exponential moving average of the embedding vectors
        # self.vq = hk.nets.VectorQuantizerEMA(
        #     embedding_dim=config.code_dim,
        #     num_embeddings=config.num_codes,
        #     commitment_cost=config.beta,
        #     decay=config.ema_decay,
        #     name="VQEMA",
        # )

        self.vq = VQEmbeddingEMA(
            epsilon=config.epsilon,
            num_codebooks=config.num_codebooks,
            num_embs=config.num_codes,
            emb_dim=16,
            num_discrete_latents=config.num_discrete_latents,
            decay=config.ema_decay,
            commitment_loss=config.beta,
        )

    def __call__(self, states: jnp.ndarray, is_training: bool = True, **kwargs):
        """IDM takes the state and next state and predicts the action
        IDM predicts the latent action (z_t) given o_t-k,..., o_t and o_t+1

        Input:
            states: (B, T, D) or (B, T, H, W, C) if state or image observations

        Output:
            vq_outputs: dict
        """
        if self.config.use_state_diff:
            # compute the difference between states
            states = states[:, 1:] - states[:, :-1]

        if self.use_transformer:
            # first embed the states with CNN backbone
            # TODO: should we consider using a patchified approach
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

        # predict latent actions
        latent_actions = self.latent_action_head(nn.gelu(state_embeds))

        if self.use_transformer:
            # combine b and t
            latent_actions = einops.rearrange(latent_actions, "b t dl -> (b t) dl")
            vq_outputs = self.vq(latent_actions, is_training=is_training)
            vq_outputs.latent_actions = latent_actions

            def reshape_fn(x):
                if x.ndim > 0:
                    return einops.rearrange(x, "(b t) ... -> b t ...", b=b)
                else:
                    return x

            # should be (b, t-1, dl)
            vq_outputs = jax.tree_util.tree_map(reshape_fn, vq_outputs)
        else:
            # compute quantized latent actions
            vq_outputs = self.vq(latent_actions, is_training=is_training)
            vq_outputs.latent_actions = latent_actions

        return vq_outputs
