from absl import logging
import einops
import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from varibad_jax.models.transformer_encoder import SARTransformerEncoder
from varibad_jax.agents.common import ActionHead


class DecisionTransformer(hk.Module):
    def __init__(
        self,
        config: ConfigDict,
        image_obs: bool,
        action_dim: int,
        is_continuous: bool = False,
        gaussian_policy: bool = False,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
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
            image_obs=image_obs,
            batch_first=config.batch_first,
            w_init=w_init,
            b_init=b_init
        )
        self.action_head = ActionHead(
            gaussian_policy=gaussian_policy,
            is_continuous=is_continuous,
            action_dim=action_dim,
            w_init=w_init,
            b_init=b_init,
        )

    def __call__(
        self,
        states: jax.Array,  # [B, T, D]
        actions: jax.Array,  # [B, T, D]
        mask: jax.Array,  # [B, T]
        rewards: jax.Array = None,  # [B, T, 1]
        prompt: jax.Array = None,
        traj_index: jax.Array = None,
        is_training: bool = True,
        **kwargs
    ) -> jax.Array:
        logging.info("inside DT")

        # [B, T*3, D]
        embeddings = self.transformer(
            states=states,
            actions=actions,
            rewards=rewards,
            mask=mask,
            prompt=prompt,
            traj_index=traj_index,
            is_training=is_training,
        )

        # reshape embeddings to [B, T, 3, D]
        num_tokens = 3 if rewards is not None else 2
        if prompt is not None:
            prompt_embedding = embeddings[:, 0]
            embeddings = embeddings[:, 1:]

        embeddings = einops.rearrange(embeddings, "b (t c) d -> b c t d", c=num_tokens)

        # predict actions from the state embedding
        if rewards is None:
            state_embed = embeddings[:, 0]
        else:
            state_embed = embeddings[:, 1]

        policy_output = self.action_head(state_embed, is_training=is_training)
        return policy_output
