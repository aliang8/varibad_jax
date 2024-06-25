from absl import logging
import haiku as hk
import dataclasses
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import einops
from ml_collections import ConfigDict
from varibad_jax.models.common import ImageEncoder


class TransformerEncoderLayer(hk.Module):
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
                nn.gelu,
                hk.Linear(self.hidden_dim, w_init=w_init, b_init=b_init),
            ]
        )

        self.ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array,
        is_training: bool = True,
    ) -> jax.Array:
        h_norm = self.ln(x)
        h_attn = self.attn_block(h_norm, h_norm, h_norm, mask=mask)
        # jax.debug.breakpoint()
        if is_training:
            h_attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_attn)
        h = x + h_attn

        h_norm = self.ln(h)
        h_dense = self.dense_block(h_norm)
        if is_training:
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
        causal: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor
        self.causal = causal

        initializer = hk.initializers.VarianceScaling(2 / self.num_layers)

        self.layers = [
            TransformerEncoderLayer(
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
        mask: jax.Array = None,  # [B, T]
        causal_mask: jax.Array = None,  # [B, T, T]
        is_training: bool = True,
    ) -> jax.Array:  # [B, T, D]
        """Transforms input embedding sequences to output embedding sequences."""
        logging.info(f"embeddings shape: {embeddings.shape}")
        B, seq_len, D = embeddings.shape

        if mask is None:
            mask = jnp.ones((B, seq_len))

        # Compute causal mask for autoregressive sequence modelling.
        mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]

        if self.causal:
            if causal_mask is None:
                causal_mask = np.tril(
                    np.ones((1, 1, seq_len, seq_len))
                )  # [B=1, H=1, T, T]

            mask = mask * causal_mask  # [B, H=1, T, T]

        h = embeddings
        for i, layer in enumerate(self.layers):
            logging.info(f"h shape: {h.shape}")
            h = layer(h, mask, is_training=is_training)

        return self.ln(h)


class TransformerDecoderLayer(hk.Module):
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
                nn.gelu,
                hk.Linear(self.hidden_dim, w_init=w_init, b_init=b_init),
            ]
        )

        self.ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, x, key, value, src_mask, tgt_mask, is_training: bool = True):
        # layer norm -> attention -> dropout -> residual connection
        h_norm = self.ln(x)
        h_attn = self.attn_block(h_norm, h_norm, h_norm, mask=tgt_mask)
        if is_training:
            h_attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_attn)

        h = x + h_attn

        h_norm = self.ln(h)
        # cross attention with encoder output -> dropout -> residual connection
        h_cross_attn = self.attn_block(h_norm, key, value, mask=src_mask)
        if is_training:
            h_cross_attn = hk.dropout(
                hk.next_rng_key(), self.dropout_rate, h_cross_attn
            )
        h = h + h_cross_attn

        h_norm = self.ln(h)
        h_dense = self.dense_block(h_norm)
        if is_training:
            h_dense = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_dense)

        h = h + h_dense
        return h


class TransformerDecoder(hk.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        attn_size: int,
        dropout_rate: float,
        widening_factor: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor

        initializer = hk.initializers.VarianceScaling(2 / self.num_layers)

        self.layers = [
            TransformerDecoderLayer(
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
        enc_out: jax.Array,  # [B, T, D]
        src_mask: jax.Array = None,  # [B, T]
        tgt_mask: jax.Array = None,  # [B, T]
        causal_mask: jax.Array = None,  # [B, T, T]
        is_training: bool = True,
    ) -> jax.Array:
        logging.info(f"embeddings shape: {embeddings.shape}")
        B, seq_len, D = embeddings.shape

        if tgt_mask is None:
            tgt_mask = jnp.ones((B, seq_len))

        # Compute causal mask for autoregressive sequence modelling.
        tgt_mask = tgt_mask[:, None, None, :]  # [B, H=1, T'=1, T]

        if causal_mask is None:
            causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len)))  # [B=1, H=1, T, T]

        tgt_mask = tgt_mask * causal_mask  # [B, H=1, T, T]

        # source mask is for the encoder output
        h = embeddings

        for i, layer in enumerate(self.layers):
            logging.info(f"h shape: {h.shape}")
            h = layer(
                x=h,
                key=enc_out,
                value=enc_out,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                is_training=is_training,
            )

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
        batch_first: bool = False,
        image_encoder_config: ConfigDict = None,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs,
    ):
        super().__init__()
        self.image_obs = image_obs
        self.batch_first = batch_first
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
            self.state_embed = ImageEncoder(**image_encoder_config, **init_kwargs)
        else:
            self.state_embed = hk.Linear(embedding_dim, **init_kwargs)
        self.action_embed = hk.Linear(embedding_dim, **init_kwargs)
        self.reward_embed = hk.Linear(embedding_dim, **init_kwargs)
        self.timestep_embed = hk.Embed(max_timesteps, embedding_dim)

        self.prompt_embed = hk.Linear(embedding_dim, **init_kwargs)
        self.traj_index_embed = hk.Embed(5, embedding_dim)

        self.embed = hk.Linear(hidden_dim, **init_kwargs)

    def __call__(
        self,
        states: jax.Array,  # [T, B, D]
        actions: jax.Array,  # [T, B, D]
        mask: jax.Array,  # [T, B]
        rewards: jax.Array = None,  # [T, B, 1]
        prompt: jax.Array = None,
        timestep: jax.Array = None,  # [T, B]
        traj_index: jax.Array = None,
        is_training: bool = True,
        **kwargs,
    ) -> jax.Array:
        """
        Implement causal transformer. Each of state, action, and reward are either embedded
        as separate tokens or treated as a single timestep.

        Reward can be optional. If not provided, then we will just have state, action sequences.
        """

        # make batch first
        if self.batch_first:
            B, T, *_ = states.shape
        else:
            T, B, *_ = states.shape
            # jax.debug.breakpoint()
            # only swap first two axes
            states = einops.rearrange(states, "t b ... -> b t ...")
            actions = einops.rearrange(actions, "t b ... -> b t ...")
            if rewards is not None:
                rewards = einops.rearrange(rewards, "t b ... -> b t ...")
            mask = einops.rearrange(mask, "t b -> b t")

        if self.image_obs:
            states = einops.rearrange(states, "b t h w c -> b t c h w")
            state_embed = self.state_embed(states, is_training=is_training)
        else:
            state_embed = self.state_embed(states)
        action_embed = self.action_embed(actions)

        if prompt is not None:
            prompt_embed = self.prompt_embed(prompt)

        if timestep is None:
            timestep = jnp.arange(T)[None].repeat(B, axis=0)

        timestep_embed = self.timestep_embed(timestep.astype(jnp.int32))

        if rewards is not None:
            reward_embed = self.reward_embed(rewards)
            reward_embed = reward_embed + timestep_embed

        # jax.debug.breakpoint()

        state_embed = state_embed + timestep_embed  # [B, T, D]
        action_embed = action_embed + timestep_embed

        # for ICL training, denote trajectory index
        if traj_index is not None:
            traj_index_embed = self.traj_index_embed(traj_index.astype(jnp.int32))
            state_embed = state_embed + traj_index_embed
            action_embed = action_embed + traj_index_embed

        if self.encode_separate:
            # stack, [B, T, D] -> [B, 3, T, D]
            # reward should be RTG if available

            if rewards is None:
                embeddings = jnp.stack([state_embed, action_embed], axis=1)
            else:
                embeddings = jnp.stack(
                    [reward_embed, state_embed, action_embed], axis=1
                )

            # make one long sequence
            embeddings = einops.rearrange(embeddings, "b c t d -> b (t c) d")

            if prompt is not None:
                embeddings = jnp.concatenate([prompt_embed, embeddings], axis=1)

            # make mask
            num_tokens = 3 if rewards is not None else 2
            mask = mask[:, None, :].repeat(num_tokens, axis=1)
            mask = einops.rearrange(mask, "b c t -> b (t c)")
            # add one for the prompt
            if prompt is not None:
                mask = jnp.concatenate([jnp.ones((B, 1)), mask], axis=1)
        else:
            embeddings = jnp.concatenate(
                [state_embed, action_embed, reward_embed], axis=-1
            )
            embeddings = nn.gelu(embeddings)
            embeddings = self.embed(embeddings)
            embeddings = nn.gelu(embeddings)

        embeddings = self.transformer(embeddings, mask, is_training=is_training)

        if not self.batch_first:
            # this is [B, T, D], need to reshape
            embeddings = einops.rearrange(embeddings, "b t d -> t b d")

        return embeddings


if __name__ == "__main__":
    # test encoder
    B = 2
    T = 5
    D = 64
    hidden_dim = 64
    num_heads = 4
    num_layers = 2
    attn_size = 32
    dropout_rate = 0.1

    @hk.transform
    def forward_encoder(x, mask):
        encoder = TransformerEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            dropout_rate=dropout_rate,
        )
        out = encoder(x, mask, is_training=True)
        print(f"out shape: {out.shape}")
        return out

    states = jnp.ones((B, T, D))
    mask = jnp.ones((B, T))

    params = forward_encoder.init(jax.random.PRNGKey(0), states, mask)

    @hk.transform
    def forward_decoder(x, enc_out, src_mask, tgt_mask):
        decoder = TransformerDecoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            dropout_rate=dropout_rate,
        )
        out = decoder(
            x, enc_out, src_mask=src_mask, tgt_mask=tgt_mask, is_training=True
        )
        print(out.shape)
        return out

    states = jnp.ones((B, T, D))
    mask = jnp.ones((B, T))
    enc_out = jnp.ones((B, T, D))
    src_mask = jnp.ones((B, 1, T, T))
    tgt_mask = jnp.ones((B, T))

    params = forward_decoder.init(
        jax.random.PRNGKey(0), states, enc_out, src_mask, tgt_mask
    )
