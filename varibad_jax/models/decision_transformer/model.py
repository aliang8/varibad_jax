import einops
import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from varibad_jax.models.transformer_encoder import SARTransformerEncoder
from varibad_jax.models.common import ImageEncoder
from varibad_jax.agents.ppo.common import ActionHead


class DecisionTransformer(hk.Module):
    def __init__(
        self,
        config: ConfigDict,
        action_dim: int,
        is_continuous: bool = False,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        batch_first: bool = True,
        **kwargs
    ):
        """
        For DT, we want each (s, a, r) to be encoded separately in a sequence.
        """
        super().__init__()
        transformer_config = ConfigDict(config.transformer_config)
        transformer_config.encode_separate = True

        self.transformer = SARTransformerEncoder(
            **transformer_config,
            image_encoder_config=config.image_encoder_config,
            image_obs=config.image_obs,
            batch_first=batch_first,
            w_init=w_init,
            b_init=b_init
        )
        self.action_head = ActionHead(is_continuous, action_dim, w_init, b_init)

    def __call__(
        self,
        states: jax.Array,  # [B, T, D]
        actions: jax.Array,  # [B, T, D]
        rewards: jax.Array,  # [B, T, 1]
        mask: jax.Array,  # [B, T]
        is_training: bool = True,
        **kwargs
    ) -> jax.Array:
        # [B, T*3, D]
        embeddings = self.transformer(
            states, actions, rewards, mask, is_training=is_training
        )

        # reshape embeddings to [B, T, 3, D]
        embeddings = einops.rearrange(embeddings, "b (t c) d -> b t c d", c=3)

        # predict actions from the state embedding
        state_embed = embeddings[:, :, 1]

        action_pred = self.action_head(state_embed)
        return action_pred
