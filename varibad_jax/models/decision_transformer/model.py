import einops
import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from varibad_jax.models.transformer_encoder import SARTransformerEncoder


class DecisionTransformer(hk.Module):
    def __init__(
        self,
        image_obs: bool,
        transformer_config: ConfigDict,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs
    ):
        """
        For DT, we want each (s, a, r) to be encoded separately in a sequence.
        """
        transformer_config = ConfigDict(transformer_config)
        transformer_config.encode_separate = True

        self.transformer = SARTransformerEncoder(
            **transformer_config, image_obs=image_obs, w_init=w_init, b_init=b_init
        )
        self.action_head = hk.Linear(1, w_init=w_init, b_init=b_init)

    def __call__(
        self,
        states: jax.Array,  # [T, B, D]
        actions: jax.Array,  # [T, B, D]
        rewards: jax.Array,  # [T, B, 1]
        mask: jax.Array,  # [T, B]
        deterministic: bool = False,
        **kwargs
    ) -> jax.Array:

        # [T*3, B, D]
        embeddings = self.transformer(
            states, actions, rewards, mask, deterministic=deterministic
        )

        # reshape embeddings to [T, 3, B, D]
        embeddings = einops.rearrange(embeddings, "t (c b) d -> t c b d", c=3)

        # predict actions from the state embedding
        state_embed = embeddings[:, 1]

        action_pred = self.action_head(state_embed)
        return action_pred
