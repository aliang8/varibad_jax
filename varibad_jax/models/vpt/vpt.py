from absl import logging
from typing import Tuple

import re
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

from varibad_jax.utils.data_utils import Batch
from varibad_jax.models.base import BaseModel
from varibad_jax.models.vpt.model import IDM
from varibad_jax.models.base import BaseAgent
from varibad_jax.agents.actor_critic import ActorCritic, ActorCriticInput
from varibad_jax.models.decision_transformer.dt import DecisionTransformerAgent


class VPT(BaseModel):
    @hk.transform_with_state
    def model(self, states, is_training=True):
        idm = IDM(
            self.config.idm,
            is_continuous=self.is_continuous,
            action_dim=self.action_dim,
        )

        # IDM predicts the ground truth action (a_t) given o_t-k, ..., o_t and o_t+1
        action_pred = idm(states, is_training=is_training)
        return action_pred

    @partial(jax.pmap, axis_name="device", static_broadcasted_argnums=(0,))
    def _init_model(self, init_key: jax.random.PRNGKey):
        t, bs = 2 + self.config.context_len, 2
        dummy_states = np.zeros((bs, t, *self.observation_shape), dtype=np.float32)
        # dummy_states = np.zeros((bs, t, 64, 64, 3), dtype=np.float32)

        params, state = self.model.init(
            init_key,
            self,
            states=dummy_states,
            is_training=True,
        )
        return params, state

    def loss_fn(
        self,
        params: hk.Params,
        state: hk.State,
        rng: jax.random.PRNGKey,
        batch: Tuple,
        is_training: bool,
    ):
        logging.info(f"vpt loss function, observations: {batch.observations.shape}")

        action_pred, state = self.model.apply(
            params,
            state,
            rng,
            self,
            states=batch.observations.astype(jnp.float32),
            is_training=is_training,
        )

        if self.config.idm.use_transformer:
            gt_actions = batch.actions[:, :-1]
        else:
            # TODO: make sure i'm predicting the right action
            gt_actions = batch.actions[:, -2]

        # action prediction
        if self.is_continuous:
            loss = optax.squared_error(action_pred.action, gt_actions)
            acc = 0.0
            # cm_plot = None
        else:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                action_pred.logits, gt_actions.squeeze().astype(jnp.int32)
            )
            acc = action_pred.action == gt_actions.squeeze()

        loss = jnp.mean(loss)
        acc = jnp.mean(acc)

        metrics = {"action_pred_loss": loss, "acc": acc}
        extras = {}
        return loss, (metrics, extras, state)


class VPTAgent(BaseAgent):
    """
    BC Agent that maps observations to actions using a pretrained IDM
    """

    @hk.transform_with_state
    def model(self, states, task=None, is_training=True, **kwargs):
        # logging.info(f"action dim: {self.config.latent_action_dim}")
        # predicts the latent action
        policy = ActorCritic(
            self.config.policy,
            is_continuous=self.is_continuous,  # we are predicting latent actions which are continuous vectors
            action_dim=self.action_dim,
        )
        policy_input = ActorCriticInput(
            state=states, task=task, is_training=is_training
        )
        policy_output = policy(policy_input, hidden_state=None)
        return policy_output

    @partial(jax.pmap, axis_name="device", static_broadcasted_argnums=(0,))
    def _init_model(self, init_key: jax.random.PRNGKey):
        t, bs = 2, 2
        dummy_state = np.zeros((bs, t, *self.observation_shape), dtype=np.float32)

        if self.config.policy.pass_task_to_policy:
            dummy_task = np.zeros((bs, t, self.task_dim), dtype=np.float32)
        else:
            dummy_task = None

        params, state = self.model.init(
            init_key,
            self,
            states=dummy_state,
            task=dummy_task,
            is_training=True,
        )
        return params, state

    def loss_fn(
        self,
        params: hk.Params,
        state: hk.State,
        rng: jax.random.PRNGKey,
        batch: Tuple,
        is_training: bool,
    ):
        logging.info(f"lam loss function, observations: {batch.observations.shape}")

        vpt_actions = batch.latent_actions.squeeze()
        policy_key, _ = jax.random.split(rng, 2)

        # if self.config.image_obs:
        #     observations = einops.rearrange(
        #         batch.observations, "b t h w c -> b t c h w"
        #     )
        # else:
        observations = batch.observations

        # only the middle observation matters here
        # predict latent action
        (action_output, _), state = self.model.apply(
            params,
            state,
            policy_key,
            self,
            states=observations.astype(jnp.float32),
            task=batch.tasks,
            is_training=is_training,
        )

        # action prediction
        if self.is_continuous:
            loss = optax.squared_error(action_output.action, vpt_actions)
            acc = 0.0
            # cm_plot = None
        else:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                action_output.logits, vpt_actions.squeeze().astype(jnp.int32)
            )
            acc = action_output.action == vpt_actions.squeeze()

        loss = jnp.mean(loss)
        acc = jnp.mean(acc)

        metrics = {"bc_loss": loss, "acc": acc}
        extras = {}
        return loss, (metrics, extras, state)


class VPTDTAgent(DecisionTransformerAgent, VPTAgent):
    def __init__(self, *args, **kwargs):
        VPTAgent.__init__(self, *args, **kwargs)

    def _init_model(self):
        return DecisionTransformerAgent._init_model(self)

    def label_trajectory_with_actions(self, rng, observations):
        b = observations.shape[0]

        # we need to unfold the observations to feed into
        # the latent action model
        if not self.vpt_cfg.model.idm.use_transformer:
            windows = []
            window_size = 2 + self.vpt_cfg.model.context_len
            for i in range(observations.shape[1] - window_size + 1):
                window = observations[:, i : i + window_size]
                windows.append(window)

            windows = jnp.stack(windows, axis=1)
            observations = einops.rearrange(windows, "b t ... -> (b t) ...")

        vpt_output, _ = self.vpt.model.apply(
            self.vpt._ts.params,
            self.vpt._ts.state,
            rng,
            self.vpt,
            states=observations,
            is_training=False,
        )
        actions = vpt_output.action

        if not self.vpt_cfg.model.idm.use_transformer:
            # need to reshape it back, squeeze because the lam predicts a single action
            actions = einops.rearrange(actions, "(b t) ... -> b t ...", b=b)

            # we're going to be missing the last action
            # just repeat it for now
            actions = jnp.concatenate([actions, actions[:, -2:-1]], axis=1)

        return actions

    def loss_fn(
        self,
        params: hk.Params,
        state: hk.State,
        rng: jax.random.PRNGKey,
        batch: Batch,
        is_training: bool,
    ):
        # observations should be B, T, ...
        logging.info(f"VPT DT loss function, observations: {batch.observations.shape}")

        vpt_key, policy_key = jax.random.split(rng, 2)

        # predict the latent action from observations with pretrained model
        actions = self.label_trajectory_with_actions(
            vpt_key, batch.observations.astype(jnp.float32)
        )
        acc = actions == batch.actions.squeeze()
        batch.actions = actions
        loss, (metrics, new_state) = DecisionTransformerAgent.loss_fn(
            self, params, state, policy_key, batch, is_training
        )
        metrics["acc"] = jnp.mean(acc)
        return loss, (metrics, new_state)

    @partial(jax.jit, static_argnames=("self", "is_training"))
    def get_action_jit(self, ts, rng, env_state, task=None, **kwargs):
        logging.info("inside get action for VPTDTAgent")
        key, action_key = jax.random.split(rng, 2)
        vpt_output, state = self.model.apply(
            ts.params, ts.state, action_key, self, env_state, task, **kwargs
        )
        return vpt_output, state
