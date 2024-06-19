from typing import Tuple, Dict
import jax
import numpy as np
import haiku as hk
import einops
import jax.numpy as jnp
from varibad_jax.models.transformer_encoder import (
    TransformerEncoder,
    TransformerDecoder,
)
from varibad_jax.models.lam.helpers import VQEmbeddingEMA
from ml_collections.config_dict import ConfigDict


class ViTIDMSingle(hk.Module):
    def __init__(
        self,
        config: ConfigDict,
        init_kwargs: Dict,
    ):
        """
        ViT architecture for implementing Image-based IDM
        """
        super().__init__()
        self.config = config
        self.init_kwargs = init_kwargs
        self.hidden_dim = config.transformer_config.hidden_dim
        self.dropout_rate = config.transformer_config.dropout_rate
        self.image_height, self.image_width = config.transformer_config.image_size
        self.patch_height, self.patch_width = config.transformer_config.patch_size

        assert self.image_height % self.patch_height == 0

        num_patches = (self.image_height // self.patch_height) * (
            self.image_width // self.patch_width
        )
        self.num_patches = num_patches

        self.to_patch_embedding = hk.Sequential(
            [
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                hk.Linear(self.hidden_dim, **init_kwargs),
            ]
        )

        # # If img_size=(84, 84), patch_size=(14, 14), then P = 84 / 14 = 6.
        # self.image_emb = hk.Conv2d(
        #     out_channels=self.hidden_dim,
        #     kernel_size=(self.patch_height, self.patch_width),
        #     stride=(self.patch_height, self.patch_width),
        #     padding="VALID",
        # )  # image_emb is now [BT x D x P x P].

        self.timestep_embed = hk.Linear(self.hidden_dim, **init_kwargs)
        self.transformer = TransformerEncoder(
            **config.transformer_config,
            causal=False,
        )

        self.vq = VQEmbeddingEMA(
            epsilon=config.epsilon,
            num_codebooks=config.num_codebooks,
            num_embs=config.num_codes,
            emb_dim=16,
            num_discrete_latents=config.num_discrete_latents,
            decay=config.ema_decay,
            commitment_loss=config.beta,
        )

    def __call__(
        self,
        states: jnp.ndarray,
        timesteps: jnp.ndarray = None,
        is_training: bool = True,
    ):
        """
        Predicts latent action from sequence of images. We patchify the images
        and pass them through a linear layer and then a bidirectional transformer encoder.

        Add patch embed and timestep embed for each image in the sequence.

        o_1, o_2, .... o_T -> [p_1, p_2, ...]

        Args:
            states: (B, T, C, H, W)
        """
        b, t = states.shape[:2]

        # the cls token is used for predicting the latent actions
        cls_token = einops.repeat(self.cls_token, "1 1 d -> b 1 d", b=b)

        # split image into patches
        patches = einops.rearrange(
            states,
            "b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)",
            p1=self.patch_height,
            p2=self.patch_width,
            t=t,
        )
        patch_emb = self.to_patch_embedding(patches)

        # reshape to [B, T * num_patches, D]
        patch_emb = einops.rearrange(patch_emb, "b t n d -> b (t n) d")

        # add cls token, [B, 1 + T * num_patches, D]
        patch_emb = jnp.concatenate([cls_token, patch_emb], axis=1)

        # repeat img_pos_enc for each timestep
        img_pos_enc = einops.repeat(self.img_pos_enc, "1 n d -> b t n d", b=b, t=t)
        img_pos_enc = einops.rearrange(img_pos_enc, "b t n d -> b (t n) d")
        img_pos_enc = jnp.concatenate([self.cls_token_pos, img_pos_enc], axis=1)
        patch_emb += img_pos_enc

        if is_training:
            patch_emb = hk.dropout(hk.next_rng_key(), self.dropout_rate, patch_emb)

        x = self.transformer(patch_emb)

        # take the cls embedding as the latent action
        latent_action = x[:, 0]

        # [BT, D]
        latent_action = einops.rearrange(latent_action, "b t d -> (b t) d")

        # apply a linear layer to the latent actions
        latent_action = hk.Linear(self.hidden_dim, **self.init_kwargs)(latent_action)

        # apply VQ to latent actions to discretize them
        vq_outputs = self.vq(latent_action)

        # reshape back to [B, T, ...]
        def reshape_obs(x):
            if x is not None and len(x.shape) > 0:
                return einops.rearrange(x, "(b t) ... -> b t ...", b=b, t=t)
            return x

        vq_outputs = jax.tree_util.tree_map(reshape_obs, vq_outputs)
        return vq_outputs


class ViTIDMSequence(ViTIDMSingle):
    def __call__(
        self,
        states: jnp.ndarray,
        timestep: jnp.ndarray = None,
        is_training: bool = True,
    ):
        """
        Predicts latent action from sequence of images. We patchify the images
        and pass them through a linear layer and then a bidirectional transformer encoder.

        Add patch embed and timestep embed for each image in the sequence.

        o_1, o_2, .... o_T -> [p_1, p_2, ...]

        Args:
            states: (B, T, H, W, C)

        Return:
            vq_outputs: Dict
                quantize: (B, T, D)
        """
        b, t = states.shape[:2]
        num_channels = states.shape[-1]

        # the cls token is used for predicting the latent actions
        # predict one latent action for each observation
        # one CLS token for each timestep
        cls_token = hk.get_parameter(
            "cls_token",
            shape=(1, t, self.hidden_dim),
            init=hk.initializers.RandomNormal(0.02),
        )
        cls_token = einops.repeat(cls_token, "1 t d -> b t 1 d", b=b, t=t)

        # split image into patches
        if self.config.patch_with_conv:
            flattened_states = einops.rearrange(states, "b t h w c -> (b t) h w c")
            patches = hk.Conv2D(
                output_channels=self.num_patches * num_channels,
                kernel_shape=(self.patch_height, self.patch_width),
                stride=(self.patch_height, self.patch_width),
                padding="VALID",
                data_format="NHWC",
            )(flattened_states)

            # reshape
            patches = einops.rearrange(
                patches,
                "(b t) h w d -> b t (h w) d",
                b=b,
                t=t,
                h=self.image_height // self.patch_height,
                w=self.image_width // self.patch_width,
            )
        else:
            patches = einops.rearrange(
                states,
                "b t (h p1) (w p2) c -> b t (h w) (p1 p2 c)",
                p1=self.patch_height,
                p2=self.patch_width,
                t=t,
            )
        # [B, T, D, # patches]
        patch_emb = self.to_patch_embedding(patches)

        # add timestep embedding, [B, T, D]
        timestep = timestep[..., None]
        timestep_embed = self.timestep_embed(timestep.astype(jnp.float32))
        timestep_embed = einops.repeat(
            timestep_embed, "b t h -> b t d h", d=self.num_patches
        )
        patch_emb += timestep_embed

        # add cls token, [B, T, 1+num_patches, D]
        patch_emb = jnp.concatenate([cls_token, patch_emb], axis=2)

        # ADD POSITION ENCODINGS
        # repeat img_pos_enc for each timestep
        img_pos_enc = hk.get_parameter(
            "img_pos_enc",
            shape=(1, self.num_patches, self.hidden_dim),
            init=hk.initializers.RandomNormal(0.02),
        )
        img_pos_enc = einops.repeat(img_pos_enc, "1 n d -> b t n d", b=b, t=t)

        cls_token_pos = hk.get_parameter(
            "cls_token_pos",
            shape=(1, 1, self.hidden_dim),
            init=hk.initializers.RandomNormal(0.02),
        )
        cls_token_pos = einops.repeat(cls_token_pos, "1 1 d -> b t 1 d", b=b, t=t)
        img_pos_enc = jnp.concatenate([cls_token_pos, img_pos_enc], axis=2)
        patch_emb += img_pos_enc

        if is_training:
            patch_emb = hk.dropout(hk.next_rng_key(), self.dropout_rate, patch_emb)

        # reshape to [B, T * (num_patches + 1), D]
        patch_emb = einops.rearrange(patch_emb, "b t n d -> b (t n) d")

        x = self.transformer(patch_emb)

        # reshape back to [B, T, num_patches + 1, D]
        x = einops.rearrange(
            x, "b (t n) d -> b t n d", b=b, t=t, n=1 + self.num_patches
        )

        # take the cls embedding for each timestep as the latent action
        # remove the last one
        # latent_action = x[:, :-1, 0]
        # [B, T, D]
        latent_action = x[:, :, 0]

        # [BT, D]
        latent_action = einops.rearrange(latent_action, "b t d -> (b t) d")

        # apply a linear layer to the latent actions
        latent_action = hk.Linear(self.hidden_dim, **self.init_kwargs)(latent_action)
        # apply activation
        latent_action = jax.nn.relu(latent_action)

        # apply VQ to latent actions to discretize them
        vq_outputs = self.vq(latent_action)

        # reshape back to [B, T, ...]
        def reshape_obs(x):
            if x is not None and len(x.shape) > 0:
                return einops.rearrange(x, "(b t) ... -> b t ...", b=b, t=t)
            return x

        vq_outputs = jax.tree_util.tree_map(reshape_obs, vq_outputs)
        vq_outputs.latent_action = latent_action
        return vq_outputs


def create_lower_triangular_block_matrix(B, n_blocks):
    """
    Create a lower triangular block matrix using a given block matrix B.

    Parameters:
    B (np.ndarray): The block matrix to be repeated.
    n_blocks (int): The number of blocks along the diagonal.

    Returns:
    np.ndarray: The lower triangular block matrix.
    """
    # Create a lower triangular matrix of size n_blocks
    L = jnp.tril(jnp.ones((n_blocks, n_blocks)))

    # Generate the block matrix using the Kronecker product
    lower_triangular_block_matrix = jnp.kron(L, B)

    return lower_triangular_block_matrix


class ViTFDM(hk.Module):
    def __init__(
        self,
        config: ConfigDict,
        init_kwargs: Dict,
    ):
        """
        ViT architecture for implementing image-based transformer FDM
        """
        super().__init__()
        self.config = config
        self.init_kwargs = init_kwargs
        self.hidden_dim = config.transformer_config.hidden_dim
        self.dropout_rate = config.transformer_config.dropout_rate
        self.image_height, self.image_width = config.transformer_config.image_size
        self.patch_height, self.patch_width = config.transformer_config.patch_size

        num_patches = (self.image_height // self.patch_height) * (
            self.image_width // self.patch_width
        )
        self.num_patches = num_patches

        self.transformer = TransformerDecoder(
            **config.transformer_config,
            causal=True,
        )
        self.to_patch_embedding = hk.Sequential(
            [
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                hk.Linear(self.hidden_dim, **init_kwargs),
            ]
        )

        self.timestep_embed = hk.Linear(self.hidden_dim, **init_kwargs)

    def __call__(
        self,
        states: jnp.ndarray,
        latent_actions: jnp.ndarray,
        timestep: jnp.ndarray = None,
        src_mask: jnp.ndarray = None,
        tgt_mask: jnp.ndarray = None,
        is_training: bool = True,
    ):
        """
        Transformer decoder to predict future states from latent action

        Args:
            states: (B, T, C, H, W)
            latent_actions: (B, T, D)
            is_training: bool
        Out:
            Reconstructed patches: (B, T, C, H, W)
        """
        # make sure the timesteps for states and latent_actions are the same
        assert states.shape[1] == latent_actions.shape[1]

        b, t = states.shape[:2]
        num_channels = states.shape[-1]

        # split image into patches
        if self.config.patch_with_conv:
            flattened_states = einops.rearrange(states, "b t h w c -> (b t) h w c")
            patches = hk.Conv2D(
                output_channels=self.num_patches * num_channels,
                kernel_shape=(self.patch_height, self.patch_width),
                stride=(self.patch_height, self.patch_width),
                padding="VALID",
                data_format="NHWC",
            )(flattened_states)

            # reshape
            patches = einops.rearrange(
                patches,
                "(b t) h w d -> b t (h w) d",
                b=b,
                t=t,
                h=self.image_height // self.patch_height,
                w=self.image_width // self.patch_width,
            )
        else:
            patches = einops.rearrange(
                states,
                "b t (h p1) (w p2) c -> b t (h w) (p1 p2 c)",
                p1=self.patch_height,
                p2=self.patch_width,
                t=t,
            )
        patch_emb = self.to_patch_embedding(patches)

        # add timestep embedding
        timestep = timestep[..., None]
        timestep_embed = self.timestep_embed(timestep.astype(jnp.float32))
        timestep_embed = einops.repeat(
            timestep_embed, "b t h -> b t n h", n=self.num_patches
        )
        patch_emb += timestep_embed

        # repeat img_pos_enc for each timestep
        img_pos_enc = hk.get_parameter(
            "img_pos_enc",
            shape=(1, self.num_patches, self.hidden_dim),
            init=hk.initializers.RandomNormal(0.02),
        )

        img_pos_enc = einops.repeat(img_pos_enc, "1 n d -> b t n d", b=b, t=t)
        patch_emb += img_pos_enc

        if is_training:
            patch_emb = hk.dropout(hk.next_rng_key(), self.dropout_rate, patch_emb)

        # reshape to [B, T * num_patches, D]
        patch_emb = einops.rearrange(patch_emb, "b t n d -> b (t n) d")

        target_len = patch_emb.shape[1]
        source_len = latent_actions.shape[1]

        # should be lower triangular block matrix
        causal_mask = create_lower_triangular_block_matrix(
            jnp.ones((self.num_patches, self.num_patches)), t
        )

        # tgt_mask = einops.repeat(
        #     tgt_mask, "d1 d2 -> b t d1 d2", b=b, t=t, d1=target_len
        # )

        # controls attention to encoder output
        src_mask = np.ones((target_len, source_len))
        # for j in range(latent_actions.shape[1]):
        #     src_mask[: (j + 1) * self.num_patches, j] = 1

        # [bs, 1, target_len, source_len]
        # source is coming from the encoder outputs
        src_mask = einops.repeat(src_mask, "tl sl -> b t tl sl", b=b, t=1)

        # [B, T * num_patches, D]
        patch_emb = self.transformer(
            embeddings=patch_emb,
            enc_out=latent_actions,
            src_mask=src_mask,
            # tgt_mask=tgt_mask,
            causal_mask=causal_mask,
            is_training=is_training,
        )

        # reshape back to [B, T, num_patches, D]
        reconstructed_patches = einops.rearrange(
            patch_emb, "b (t n) d -> b t n d", b=b, t=t, n=self.num_patches
        )

        # apply linear layer to decode
        # reconstruct patches of the next image
        reconstructed_patches = hk.Linear(
            self.patch_height * self.patch_width * num_channels, **self.init_kwargs
        )(reconstructed_patches)

        # reshape back to original
        reconstructed_patches = einops.rearrange(
            reconstructed_patches,
            "b t (p1 p2) (h w c) -> b t c (h p1) (w p2)",
            h=self.patch_height,
            w=self.patch_width,
            c=num_channels,
            p1=self.patch_height,
            p2=self.patch_width,
        )

        return reconstructed_patches


if __name__ == "__main__":
    import jax
    from ml_collections.config_dict import ConfigDict

    config = ConfigDict(
        dict(
            transformer_config=dict(
                image_size=(64, 64),
                patch_size=(8, 8),
                num_channels=3,
                hidden_dim=128,
                num_heads=4,
                num_layers=4,
                attn_size=2,
                dropout_rate=0.1,
            ),
            epsilon=1e-5,
            num_codebooks=2,
            num_codes=6,
            num_discrete_latents=4,
            ema_decay=0.99,
            beta=0.25,
        )
    )
    init_kwargs = {"w_init": hk.initializers.VarianceScaling(scale=2.0)}

    states = jnp.ones((1, 10, 64, 64, 3))

    @hk.transform
    def forward_single(x):
        model = ViTIDMSingle(
            config=config,
            init_kwargs=init_kwargs,
        )
        out = model(x)
        print(out.quantize.shape)

        # assert out.quantize.shape == (1, config.transformer_config.hidden_dim)
        return out

    @hk.transform_with_state
    def forward_sequence(x):
        model = ViTIDMSequence(
            config=config,
            init_kwargs=init_kwargs,
        )
        out = model(x)
        print(out.quantize.shape)
        # assert out.shape == (1, 10, config.transformer_config.hidden_dim)
        return out

    # params = forward_single.init(jax.random.PRNGKey(42), states)
    params, state = forward_sequence.init(jax.random.PRNGKey(42), x=states)

    @hk.transform
    def forward_fdm(x, enc_out, src_mask=None, tgt_mask=None):
        model = ViTFDM(
            config=config,
            init_kwargs=init_kwargs,
        )
        out = model(x, enc_out, src_mask, tgt_mask)
        print(out.shape)
        return out

    # # block_mat = lower_triangular_block_matrix(10, 2)
    # block_mat = lower_triangular_block_matrix(640, 10, 1)
    # print(block_mat)

    params = forward_fdm.init(
        jax.random.PRNGKey(42),
        x=states,
        enc_out=jnp.ones((1, 10, config.transformer_config.hidden_dim)),
        # src_mask=jnp.ones((1, 10)),
        # tgt_mask=jnp.ones((1, 10)),
    )
