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
            config=self.config,
            image_obs=self.config.image_obs,
            is_continuous=self.is_continuous,
            action_dim=self.action_dim,
        )
        return model(**kwargs)

    def _init_model(self):
        t, bs = 2, 2
        dummy_states = np.zeros((bs, t, *self.observation_shape), dtype=np.float32)
        dummy_actions = np.zeros((bs, t, self.input_action_dim))

        if self.config.use_rtg:
            dummy_rewards = np.zeros((bs, t, 1))
        else:
            dummy_rewards = None

        dummy_mask = np.ones((bs, t))

        if self.config.task_conditioning:
            dummy_prompt = np.zeros((bs, 1, self.task_dim))
        else:
            dummy_prompt = None

        self._params, self._state = self.model.init(
            self._init_key,
            self,
            states=dummy_states,
            actions=dummy_actions,
            rewards=dummy_rewards,
            mask=dummy_mask,
            prompt=dummy_prompt,
            is_training=True,
        )

    def loss_fn(
        self, params: hk.Params, state: hk.State, rng: jax.random.PRNGKey, batch: Tuple
    ):
        observations = batch.observations
        actions = batch.actions
        rewards = batch.rewards

        if self.config.task_conditioning:
            prompt = batch.tasks[:, 0:1]
        else:
            prompt = None

        # [B, T, 1]
        if len(actions.shape) == 2:
            actions = jnp.expand_dims(actions, axis=-1)

        # [B, T]
        mask = jnp.ones_like(rewards)

        # [B, T, 1]
        if self.config.use_rtg:
            if len(rewards.shape) == 2:
                rewards = jnp.expand_dims(rewards, axis=-1)
        else:
            rewards = None

        policy_output, new_state = self.model.apply(
            params,
            state,
            rng,
            self,
            states=observations.astype(jnp.float32),
            actions=actions,
            rewards=rewards,
            mask=mask,
            prompt=prompt,
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
            # loss *= batch.mask
            # loss = loss.sum() / batch.mask.sum()
        loss = jnp.mean(loss)

        metrics = {"bc_loss": loss, "entropy": entropy}

        return loss, (metrics, new_state)
