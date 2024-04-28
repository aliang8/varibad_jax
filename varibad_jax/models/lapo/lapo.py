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
import jax.numpy as jnp
from absl import logging
from functools import partial
from pathlib import Path
from ml_collections.config_dict import ConfigDict

from varibad_jax.models.base import BaseModel, BaseAgent
from varibad_jax.models.lapo.model import LatentFDM, LatentActionIDM
from varibad_jax.agents.actor_critic import ActorCritic
from varibad_jax.agents.common import ActionHead


@chex.dataclass
class LAPOOutput:
    next_obs_pred: jnp.ndarray
    vq_loss: jnp.ndarray
    quantize: jnp.ndarray


class LAPOModel(BaseModel):
    @hk.transform_with_state
    def model(self, states, is_training=True):
        idm = LatentActionIDM(self.config.idm)
        fdm = LatentFDM(self.config.fdm)

        # states is [B, 3, H, W, C]
        # the 3 is for o_t-1, o_t, and o_t+1

        # IDM predicts the latent action (z_t) given o_t-1, o_t and o_t+1
        idm_output = idm(states, is_training=is_training)
        # [B, D]
        latent_action = idm_output["quantize"]
        logging.info(f"latent_action: {latent_action.shape}")

        # FDM predicts o_t+1 given o_t-1, o_t and z_t
        prev_states = states[:, :-1]

        # [B, C, H, W]
        next_state_pred = fdm(prev_states, latent_action, is_training=is_training)

        # import ipdb

        # ipdb.set_trace()
        return LAPOOutput(
            next_obs_pred=next_state_pred,
            quantize=idm_output["quantize"],
            vq_loss=idm_output["loss"],
        )

    def _init_model(self):
        t, bs = 3, 2
        dummy_states = np.zeros((bs, t, *self.observation_shape), dtype=np.float32)

        self._params, self._state = self.model.init(
            self._init_key,
            self,
            states=dummy_states,
            is_training=True,
        )

    def loss_fn(
        self, params: hk.Params, state: hk.State, rng: jax.random.PRNGKey, batch: Tuple
    ):
        observations, actions, rewards = batch
        logging.info(f"lapo loss function, observations: {observations.shape}")

        lapo_output, state = self.model.apply(
            params,
            state,
            rng,
            self,
            states=observations.astype(jnp.float32),
            is_training=True,
        )

        # loss for IDM is VQVAE loss
        # loss for FDM is reconstruction loss
        next_obs_pred = lapo_output.next_obs_pred
        # make sure same format, HWC
        next_obs_pred = einops.rearrange(next_obs_pred, "b c h w -> b h w c")
        gt_next_obs = observations[:, -1]
        recon_loss = optax.squared_error(next_obs_pred, gt_next_obs)
        recon_loss = jnp.mean(recon_loss)

        metrics = {
            "recon_loss": recon_loss,
            "vq_loss": lapo_output.vq_loss,
            "total_loss": recon_loss + lapo_output.vq_loss,
        }

        loss = lapo_output.vq_loss + recon_loss
        return loss, (metrics, state)


class ActionDecoder(BaseModel):
    """
    Model that decodes latent actions to ground truth actions. Use in conjunction with the
    IDM model learned with LAPO.
    """

    def model(self, latent_actions, is_training=True):
        # some liner layers
        action_head = ActionHead(
            gaussian_policy=False,
            is_continuous=self.is_continuous,
            action_dim=self.action_dim,
        )
        output = action_head(latent_actions, is_training=is_training)
        return output

    def _init_model(self):
        bs = 2
        dummy_latent_actions = np.zeros(
            (bs, self.config.latent_action_dim), dtype=np.float32
        )
        self._params, self._state = self.model.init(
            self._init_key,
            self,
            latent_actions=dummy_latent_actions,
            is_training=True,
        )


class LAPOAgent(BaseAgent):
    """
    BC Agent that maps observations to latent actions using a pretrained IDM
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # load pretrained IDM/FDM for predicting the latent actions from observations
        config = Path(self.config.policy.lapo_model_ckpt) / "config.json"
        with open(config, "r") as f:
            self.lapo_model_cfg = ConfigDict(json.load(f))

        # TODO(anthony): hard-coding the checkpoint now, fix this
        self.lapo_model = LAPOModel(
            self.lapo_model_cfg.policy,
            key=self._init_key,
            observation_shape=self.observation_shape,
            action_dim=self.action_dim,
            input_action_dim=self.input_action_dim,
            continuous_actions=self.is_continuous,
            load_from_ckpt=True,
            ckpt_file=Path(self.config.policy.lapo_model_ckpt)
            / "model_ckpts"
            / "ckpt_400.pkl",
        )

    @hk.transform_with_state
    def model(self, states, is_training=True):
        # predicts the latent action
        policy = ActorCritic(
            self.config.policy,
            is_continuous=True,  # we are predicting latent actions which are continuous vectors
            action_dim=self.config.policy.latent_action_dim,
        )
        policy_output = policy(states, is_training=is_training)
        return policy_output

    def _init_model(self):
        t, bs = 3, 2
        dummy_state = np.zeros((bs, t, *self.observation_shape), dtype=np.float32)
        dummy_state = einops.rearrange(dummy_state, "b t h w c -> b t c h w")
        self._params, self._state = self.model.init(
            self._init_key,
            self,
            states=dummy_state,
            is_training=True,
        )

    def loss_fn(
        self, params: hk.Params, state: hk.State, rng: jax.random.PRNGKey, batch: Tuple
    ):
        observations, actions, rewards = batch
        logging.info(f"lapo loss function, observations: {observations.shape}")

        lapo_key, policy_key = jax.random.split(rng, 2)

        lapo_output, _ = self.lapo_model.model.apply(
            self.lapo_model._params,
            self.lapo_model._state,
            lapo_key,
            self.lapo_model,
            states=observations.astype(jnp.float32),
            is_training=False,
        )
        latent_action = lapo_output.quantize

        observations = einops.rearrange(observations, "b t h w c -> b t c h w")

        # only the middle observation matters here
        # not a recurrent policy
        policy_output, state = self.model.apply(
            params,
            state,
            policy_key,
            self,
            states=observations[:, 1].astype(jnp.float32),
            is_training=True,
        )

        # bc loss
        loss = optax.squared_error(policy_output.action, latent_action)
        loss = jnp.mean(loss)

        metrics = {
            "bc_loss": loss,
        }
        return loss, (metrics, state)
