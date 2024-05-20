from typing import Tuple
import einops
import jax
import numpy as np
import haiku as hk
import optax
import jax.numpy as jnp
from absl import logging
from functools import partial

from varibad_jax.utils.data_utils import Batch
from varibad_jax.models.base import BaseAgent
from varibad_jax.models.decision_transformer.model import DecisionTransformer
from varibad_jax.models.lam.lam import LatentActionBaseAgent


class DecisionTransformerAgent(BaseAgent):
    @hk.transform_with_state
    def model(self, *args, **kwargs):
        model = DecisionTransformer(
            config=self.config.policy,
            image_obs=self.config.image_obs,
            is_continuous=self.is_continuous,
            action_dim=self.action_dim,
        )
        return model(**kwargs)

    def _init_model(self):
        t, bs = 2, 2
        dummy_states = np.zeros((bs, t, *self.observation_shape), dtype=np.float32)
        dummy_actions = np.zeros((bs, t, self.input_action_dim))

        if self.config.policy.use_rtg:
            dummy_rewards = np.zeros((bs, t, 1))
        else:
            dummy_rewards = None

        dummy_mask = np.ones((bs, t))

        if self.config.policy.task_conditioning:
            dummy_prompt = np.zeros((bs, 1, self.task_dim))
        else:
            dummy_prompt = None

        dummy_traj_index = np.zeros((bs, t))

        params, state = self.model.init(
            self._init_key,
            self,
            states=dummy_states,
            actions=dummy_actions,
            rewards=dummy_rewards,
            mask=dummy_mask,
            prompt=dummy_prompt,
            traj_index=dummy_traj_index,
            is_training=True,
        )
        return params, state

    def loss_fn(
        self, params: hk.Params, state: hk.State, rng: jax.random.PRNGKey, batch: Tuple
    ):
        observations = batch.observations
        actions = batch.actions
        rewards = batch.rewards

        if self.config.policy.task_conditioning:
            prompt = batch.tasks[:, 0:1]
        else:
            prompt = None

        # [B, T, 1]
        if len(actions.shape) == 2:
            actions = jnp.expand_dims(actions, axis=-1)

        # [B, T]
        mask = jnp.ones_like(rewards)

        # [B, T, 1]
        if self.config.policy.use_rtg:
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
            traj_index=batch.traj_index,
            is_training=True,
        )

        if policy_output.entropy is not None:
            entropy = policy_output.entropy
            entropy = jnp.mean(entropy)
        else:
            entropy = 0.0

        if self.is_continuous:
            action_preds = policy_output.action
        else:
            action_preds = policy_output.logits

        if self.is_continuous:
            # compute MSE loss
            loss = optax.squared_error(action_preds, actions.squeeze())
            acc = 0.0
            loss *= batch.mask[..., jnp.newaxis]
            loss = loss.sum() / batch.mask.sum()
        else:
            # compute cross entropy with logits
            loss = optax.softmax_cross_entropy_with_integer_labels(
                action_preds, actions.squeeze(axis=-1).astype(jnp.int32)
            )
            loss *= batch.mask
            loss = loss.sum() / batch.mask.sum()

            acc = policy_output.action == actions.squeeze(axis=-1)
            acc *= batch.mask
            acc = acc.sum() / batch.mask.sum()

        metrics = {"bc_loss": loss, "entropy": entropy, "decoded_acc": acc}

        return loss, (metrics, new_state)


class LatentDTAgent(LatentActionBaseAgent, DecisionTransformerAgent):
    def __init__(self, *args, **kwargs):
        LatentActionBaseAgent.__init__(self, *args, **kwargs)

    def _init_model(self):
        return DecisionTransformerAgent._init_model(self)

    def label_trajectory_with_actions(self, rng, observations):
        b = observations.shape[0]

        # we need to unfold the observations to feed into
        # the latent action model
        if not self.lam_cfg.model.idm.use_transformer:
            windows = []
            window_size = 2 + self.lam_cfg.model.context_len
            for i in range(observations.shape[1] - window_size + 1):
                window = observations[:, i : i + window_size]
                windows.append(window)

            windows = jnp.stack(windows, axis=1)
            observations = einops.rearrange(windows, "b t ... -> (b t) ...")

        # import ipdb

        # ipdb.set_trace()

        lam_output, _ = self.lam.model.apply(
            self.lam._ts.params,
            self.lam._ts.state,
            rng,
            self.lam,
            states=observations,
            is_training=False,
        )

        latent_actions = lam_output.latent_actions

        if not self.lam_cfg.model.idm.use_transformer:
            # need to reshape it back, squeeze because the lam predicts a single action
            latent_actions = einops.rearrange(
                latent_actions, "(b t) ... -> b t ...", b=b
            )

            # we're going to be missing the last action
            # just repeat it for now
            latent_actions = jnp.concatenate(
                [latent_actions, latent_actions[:, -2:-1]], axis=1
            )

        return latent_actions

    def loss_fn(
        self, params: hk.Params, state: hk.State, rng: jax.random.PRNGKey, batch: Batch
    ):
        # observations should be B, T, ...
        logging.info(
            f"latent DT loss function, observations: {batch.observations.shape}"
        )

        lam_key, policy_key = jax.random.split(rng, 2)

        # predict the latent action from observations with pretrained model
        latent_actions = self.label_trajectory_with_actions(
            lam_key, batch.observations.astype(jnp.float32)
        )
        batch.actions = latent_actions
        return DecisionTransformerAgent.loss_fn(self, params, state, policy_key, batch)

    @partial(jax.jit, static_argnames=("self", "is_training"))
    def get_action_jit(self, ts, rng, env_state, task=None, **kwargs):
        logging.info("inside get action for LatentActionAgent")
        la_key, decoder_key = jax.random.split(rng, 2)

        # if self.config.image_obs:
        #     env_state = einops.rearrange(env_state, "b h w c -> b c h w")

        # predict latent action and then use pretrained action decoder
        la_output, state = self.model.apply(
            ts.params, ts.state, la_key, self, env_state, task, **kwargs
        )
        latent_actions = la_output.action

        # decode latent action
        action_output, _ = self.latent_action_decoder.model.apply(
            self.latent_action_decoder._ts.params,
            self.latent_action_decoder._ts.state,
            decoder_key,
            self.latent_action_decoder,
            latent_actions=latent_actions,
            is_training=False,
        )
        action_output.latent_action = latent_actions
        return action_output, state
