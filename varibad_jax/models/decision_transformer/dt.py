from typing import Tuple
import jax
import numpy as np
import haiku as hk
import optax
import jax.numpy as jnp
from absl import logging
from functools import partial

from varibad_jax.models.base import BaseAgent
from varibad_jax.models.decision_transformer.model import DecisionTransformer


class DecisionTransformerAgent(BaseAgent):
    @hk.transform_with_state
    def model(self, *args, **kwargs):
        model = DecisionTransformer(
            config=self.config.policy,
            is_continuous=self.is_continuous,
            action_dim=self.action_dim,
        )
        return model(**kwargs)

    def _init_model(self):
        t, bs = 2, 2
        dummy_states = np.zeros((t, bs, *self.observation_shape), dtype=np.float32)
        dummy_actions = np.zeros((t, bs, self.input_action_dim))
        dummy_rewards = np.zeros((t, bs, 1))
        dummy_mask = np.ones((t, bs))

        self._params, self._state = self.model.init(
            self._key,
            self,
            states=dummy_states,
            actions=dummy_actions,
            rewards=dummy_rewards,
            mask=dummy_mask,
            is_training=True,
        )

    def loss_fn(
        self, params: hk.Params, state: hk.State, batch, rng: jax.random.PRNGKey
    ):
        # trajectory level
        # observations: [B, T, *_]
        observations, actions, rewards = batch

        # [B, T, 1]
        if len(actions.shape) == 2:
            actions = jnp.expand_dims(actions, axis=-1)

        # [B, T]
        mask = jnp.ones_like(rewards)

        # [B, T, 1]
        if len(rewards.shape) == 2:
            rewards = jnp.expand_dims(rewards, axis=-1)

        policy_output, new_state = self.model.apply(
            params,
            state,
            rng,
            self,
            states=observations.astype(jnp.float32),
            actions=actions,
            rewards=rewards,
            mask=mask,
            is_training=True,
        )

        entropy = policy_output.entropy
        entropy = jnp.mean(entropy)

        action_preds = policy_output.logits

        if self.is_continuous:
            # compute MSE loss
            loss = optax.squared_error(action_preds, actions.squeeze(axis=-1))
        else:
            # compute cross entropy with logits
            loss = optax.softmax_cross_entropy_with_integer_labels(
                action_preds, actions.squeeze(axis=-1).astype(jnp.int32)
            )

        loss = jnp.mean(loss)
        metrics = {"bc_loss": loss, "entropy": entropy}

        return loss, (metrics, new_state)
