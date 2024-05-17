from absl import logging
from typing import Tuple

import jax
import json
import chex
import numpy as np
import haiku as hk
import optax
import einops
import pickle
import flax.linen as nn
import jax.numpy as jnp
from absl import logging
from functools import partial
from pathlib import Path
from ml_collections.config_dict import ConfigDict

from varibad_jax.models.base import BaseAgent
from varibad_jax.agents.actor_critic import ActorCritic, ActorCriticInput


class BCAgent(BaseAgent):
    """
    BC Agent
    """

    @hk.transform_with_state
    def model(
        self,
        states: jnp.ndarray,
        task: jnp.ndarray = None,
        is_training: bool = True,
        **kwargs,
    ):
        # predicts the latent action
        policy = ActorCritic(
            self.config.policy,
            is_continuous=self.is_continuous,
            action_dim=self.action_dim,
        )
        policy_input = ActorCriticInput(
            state=states, task=task, is_training=is_training
        )
        policy_output = policy(policy_input, hidden_state=None)
        return policy_output

    def _init_model(self):
        t, bs = 2, 2
        dummy_state = np.zeros((bs, t, *self.observation_shape), dtype=np.float32)

        if self.config.policy.pass_task_to_policy:
            dummy_task = np.zeros((bs, t, self.task_dim), dtype=np.float32)
        else:
            dummy_task = None

        params, state = self.model.init(
            self._init_key,
            self,
            states=dummy_state,
            task=dummy_task,
            is_training=True,
        )
        return params, state

    def loss_fn(
        self, params: hk.Params, state: hk.State, rng: jax.random.PRNGKey, batch: Tuple
    ):
        logging.info(f"bc loss function, observations: {batch.observations.shape}")

        rng, policy_key = jax.random.split(rng, 2)

        # predict latent action
        (action_output, _), state = self.model.apply(
            params,
            state,
            policy_key,
            self,
            states=batch.observations.astype(jnp.float32),
            task=batch.tasks,
            is_training=True,
        )

        gt_actions = batch.actions
        if self.is_continuous:
            loss = optax.squared_error(action_output.action, gt_actions)
            acc = 0.0
        else:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                action_output.logits, gt_actions.squeeze().astype(jnp.int32)
            )
            acc = action_output.action == gt_actions.squeeze()

        loss = jnp.mean(loss)
        acc = jnp.mean(acc)

        metrics = {"bc_loss": loss, "acc": acc}
        return loss, (metrics, state)
