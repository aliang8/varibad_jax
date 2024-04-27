from absl import logging
from typing import Tuple

import jax
import chex
import numpy as np
import haiku as hk
import optax
import einops
import jax.numpy as jnp
from absl import logging
from functools import partial

from varibad_jax.models.base import BaseModel
from varibad_jax.models.lapo.model import LatentFDM, LatentActionIDM


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
            self._key,
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
        }

        loss = lapo_output.vq_loss + recon_loss
        return loss, (metrics, state)
