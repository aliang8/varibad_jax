import haiku as hk
import dataclasses
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import einops
from ml_collections import ConfigDict
from varibad_jax.models.common import ImageEncoder


class TransformerLayer(hk.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_size: int,
        dropout_rate: float,
        widening_factor: int = 4,
        w_init: hk.initializers.Initializer = hk.initializers.VarianceScaling(1.0),
        b_init: hk.initializers.Initializer = hk.initializers.Constant(0.0),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_size = attn_size
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor

        self.attn_block = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.attn_size,
            model_size=hidden_dim,
            w_init=w_init,
            with_bias=True,
        )

        self.dense_block = hk.Sequential(
            [
                hk.Linear(self.hidden_dim * self.widening_factor, w_init=w_init),
                jax.nn.gelu,
                hk.Linear(self.hidden_dim, w_init=w_init),
            ]
        )

        self.ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array,
        deterministic: bool = False,
    ) -> jax.Array:
        h_norm = self.ln(x)
        h_attn = self.attn_block(h_norm, h_norm, h_norm, mask=mask)
        # jax.debug.breakpoint()
        if not deterministic:
            h_attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_attn)
        h = x + h_attn

        h_norm = self.ln(h)
        h_dense = self.dense_block(h_norm)
        if not deterministic:
            h_dense = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_dense)
        h = h + h_dense
        return h


class TransformerEncoder(hk.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        attn_size: int,
        dropout_rate: float,
        widening_factor: int = 4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor

        initializer = hk.initializers.VarianceScaling(2 / self.num_layers)

        self.layers = [
            TransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                attn_size=attn_size,
                dropout_rate=dropout_rate,
                widening_factor=widening_factor,
                w_init=initializer,
            )
            for _ in range(num_layers)
        ]

        self.ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(
        self,
        embeddings: jax.Array,  # [B, T, D]
        mask: jax.Array,  # [B, T]
        deterministic: bool = False,
    ) -> jax.Array:  # [B, T, D]
        """Transforms input embedding sequences to output embedding sequences."""
        _, seq_len, D = embeddings.shape

        # Compute causal mask for autoregressive sequence modelling.
        mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]
        causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len)))  # [B=1, H=1, T, T]
        mask = mask * causal_mask  # [B, H=1, T, T]

        h = embeddings
        for layer in self.layers:
            h = layer(h, mask, deterministic=deterministic)

        return self.ln(h)


class SARTransformerEncoder(hk.Module):
    def __init__(
        self,
        image_obs: bool,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        attn_size: int,
        dropout_rate: float,
        widening_factor: int = 4,
        max_timesteps: int = 1000,
        encode_separate: bool = False,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs
    ):
        super().__init__()
        self.image_obs = image_obs
        self.encode_separate = encode_separate
        self.transformer = TransformerEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            dropout_rate=dropout_rate,
            widening_factor=widening_factor,
        )

        init_kwargs = dict(w_init=w_init, b_init=b_init)

        if self.image_obs:
            self.state_embed = ImageEncoder(embedding_dim, **init_kwargs)
        else:
            self.state_embed = hk.Linear(embedding_dim, **init_kwargs)
        self.action_embed = hk.Linear(embedding_dim, **init_kwargs)
        self.reward_embed = hk.Linear(embedding_dim, **init_kwargs)
        self.timestep_embed = hk.Embed(max_timesteps, embedding_dim)

        self.embed = hk.Linear(hidden_dim, **init_kwargs)

    def __call__(
        self,
        states: jax.Array,  # [T, B, D]
        actions: jax.Array,  # [T, B, D]
        rewards: jax.Array,  # [T, B, 1]
        mask: jax.Array,  # [T, B]
        deterministic: bool = False,
        **kwargs
    ) -> jax.Array:
        # make batch first
        T, B, D = states.shape

        # jax.debug.breakpoint()
        states = states.transpose(1, 0, 2)
        actions = actions.transpose(1, 0, 2)
        rewards = rewards.transpose(1, 0, 2)
        mask = mask.transpose(1, 0)

        state_embed = self.state_embed(states)
        action_embed = self.action_embed(actions)
        reward_embed = self.reward_embed(rewards)
        timestep_embed = self.timestep_embed(jnp.arange(T)[None].repeat(B, axis=0))
        # jax.debug.breakpoint()

        state_embed = state_embed + timestep_embed  # [B, T, D]
        action_embed = action_embed + timestep_embed
        reward_embed = reward_embed + timestep_embed

        if self.encode_separate:
            # stack, [B, T, D] -> [B, 3, T, D]
            # reward should be RTG
            embeddings = jnp.stack([reward_embed, state_embed, action_embed], axis=1)

            # make one long sequence
            embeddings = einops.rearrange(embeddings, "b c t d -> b (c t) d")

            # make mask
            mask = mask[:, None, :].repeat(3, axis=1)
            mask = einops.rearrange(mask, "b c t -> b (c t)")
        else:
            embeddings = jnp.concatenate(
                [state_embed, action_embed, reward_embed], axis=-1
            )
            embeddings = nn.gelu(embeddings)
            embeddings = self.embed(embeddings)
            embeddings = nn.gelu(embeddings)

        embeddings = self.transformer(embeddings, mask, deterministic=deterministic)

        # this is [B, T, D], need to reshape
        embeddings = einops.rearrange(embeddings, "b t d -> t b d")
        return embeddings
