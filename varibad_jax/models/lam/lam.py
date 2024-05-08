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

from varibad_jax.models.base import BaseModel, BaseAgent
from varibad_jax.agents.actor_critic import ActorCritic
from varibad_jax.models.lam.model import LatentFDM, LatentActionIDM
from varibad_jax.agents.common import ActionHead


@partial(jax.jit, static_argnames=("n"))
def confusion_matrix(labels, predictions, n):
    cm, _ = jax.lax.scan(
        lambda carry, pair: (carry.at[pair].add(1), None),
        jnp.zeros((n, n), dtype=jnp.uint32),
        (labels, predictions),
    )
    return cm


@chex.dataclass
class LAMOutput:
    next_obs_pred: jnp.ndarray
    vq_loss: jnp.ndarray
    quantize: jnp.ndarray
    perplexity: jnp.ndarray
    encoding_indices: jnp.ndarray
    latent_actions: jnp.ndarray


def breakpoint_if_cond(x):
    # is_finite = jnp.isfinite(x).all()
    stop = x < 0.00001
    # stop = x > 0.87

    def true_fn(x):
        jax.debug.breakpoint()

    def false_fn(x):
        pass

    jax.lax.cond(stop, true_fn, false_fn, x)


class LatentActionModel(BaseModel):
    @hk.transform_with_state
    def model(self, states, is_training=True):
        idm = LatentActionIDM(self.config.idm)
        fdm = LatentFDM(self.config.fdm, state_dim=self.observation_shape[-1])

        # IDM predicts the latent action (z_t) given o_t-k, ..., o_t and o_t+1
        idm_output = idm(states, is_training=is_training)

        # FDM predicts o_t+1 given o_t-1, o_t and z_t
        context = states[:, :-1]

        # [B, C, H, W] or [B, T, C, H, W]
        next_state_pred = fdm(context, idm_output["quantize"], is_training=is_training)
        logging.info(f"next_state_pred: {next_state_pred.shape}")

        return LAMOutput(
            next_obs_pred=next_state_pred,
            vq_loss=idm_output["loss"],
            quantize=idm_output["quantize"],
            perplexity=idm_output["perplexity"],
            encoding_indices=idm_output["encoding_indices"],
            latent_actions=idm_output["latent_actions"],
        )

    def _init_model(self):
        t, bs = 2 + self.config.context_len, 2
        dummy_states = np.zeros((bs, t, *self.observation_shape), dtype=np.float32)
        # dummy_states = np.zeros((bs, t, 64, 64, 3), dtype=np.float32)

        self._params, self._state = self.model.init(
            self._init_key,
            self,
            states=dummy_states,
            is_training=True,
        )

    def loss_fn(
        self, params: hk.Params, state: hk.State, rng: jax.random.PRNGKey, batch: Tuple
    ):
        logging.info(f"lam loss function, observations: {batch.observations.shape}")

        lam_output, state = self.model.apply(
            params,
            state,
            rng,
            self,
            states=batch.observations.astype(jnp.float32),
            is_training=True,
        )

        # loss for IDM is VQVAE loss + recon
        # loss for FDM is reconstruction loss
        next_obs_pred = lam_output.next_obs_pred

        if self.config.idm.image_obs:
            # make sure same format as the ground truth, HWC
            if self.config.idm.use_transformer:
                # this should have T-1 predictions where T is our sequence length
                next_obs_pred = einops.rearrange(
                    next_obs_pred, "b t c h w -> b t h w c"
                )
                gt_next_obs = batch.observations[:, 1:]
                mask = batch.mask[:, 1:]

                recon_loss = optax.squared_error(next_obs_pred, gt_next_obs)
                recon_loss = jnp.mean(recon_loss, axis=(2, 3, 4))
                recon_loss *= mask
                recon_loss = jnp.sum(recon_loss) / jnp.sum(mask)
            else:
                next_obs_pred = einops.rearrange(next_obs_pred, "b c h w -> b h w c")
                gt_next_obs = batch.observations[:, -1]
                recon_loss = optax.squared_error(next_obs_pred, gt_next_obs)
                recon_loss = jnp.mean(recon_loss)
                # jax.debug.breakpoint()

                # breakpoint_if_cond(recon_loss)
        else:
            recon_loss = optax.squared_error(next_obs_pred, batch.observations[:, -1])
            recon_loss = jnp.mean(recon_loss)
            # breakpoint_if_cond(recon_loss)

        metrics = {
            "recon_loss": recon_loss,
            "vq_loss": lam_output.vq_loss,
            "perplexity": lam_output.perplexity,
            "total_loss": recon_loss + lam_output.vq_loss,
        }
        loss = self.config.beta_loss_weight * lam_output.vq_loss + recon_loss
        return loss, (metrics, state)


class LatentActionDecoder(BaseModel):
    """
    Model that decodes latent actions to ground truth actions. Use in conjunction with the
    IDM model learned with LAPO.
    """

    def __init__(self, lam: BaseModel = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if lam is None:
            # load pretrained latent action model
            config = Path(self.config.lam_ckpt) / "config.json"
            with open(config, "r") as f:
                self.lam_cfg = ConfigDict(json.load(f))

            self.lam = LatentActionModel(
                self.lam_cfg.model,  # TODO: fix me
                key=self._init_key,
                observation_shape=self.observation_shape,
                action_dim=self.action_dim,
                input_action_dim=self.input_action_dim,
                continuous_actions=self.is_continuous,
                load_from_ckpt=True,
                ckpt_dir=Path(self.config.lam_ckpt),
            )
        else:
            self.lam = lam

    @hk.transform_with_state
    def model(self, latent_actions, is_training=True):
        logging.info("inside action decoder")
        x = hk.nets.MLP(
            self.config.mlp_layer_sizes,
            activation=nn.gelu,
            activate_final=True,
            **self.init_kwargs,
        )(latent_actions)

        # some linear layers
        action_head = ActionHead(
            gaussian_policy=False,
            is_continuous=self.is_continuous,
            action_dim=self.action_dim,
        )
        output = action_head(x, is_training=is_training)
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

    def loss_fn(
        self, params: hk.Params, state: hk.State, rng: jax.random.PRNGKey, batch
    ):
        observations = batch.observations

        agent_key, decoder_key = jax.random.split(rng, 2)

        model_output, _ = self.lam.model.apply(
            self.lam._params,
            self.lam._state,
            agent_key,
            self.lam,
            states=observations.astype(jnp.float32),
            is_training=False,
        )

        obs_pred = model_output.next_obs_pred
        if self.lam_cfg.model.idm.image_obs:
            obs_pred = einops.rearrange(obs_pred, "b c h w -> b h w c")

        gt_next_obs = observations[:, -1]
        recon_err = optax.squared_error(obs_pred, gt_next_obs)
        recon_err = jnp.mean(recon_err)

        latent_actions = model_output.latent_actions

        # decode latent action to ground truth action
        action_pred, state = self.model.apply(
            params,
            state,
            decoder_key,
            self,
            latent_actions=latent_actions,
            is_training=True,
        )

        if self.lam_cfg.model.idm.use_transformer:
            gt_actions = batch.actions[:, :-1]
        else:
            # TODO: make sure i'm predicting the right action
            gt_actions = batch.actions[:, -2]

        if self.is_continuous:
            loss = optax.squared_error(action_pred.action, gt_actions)
            acc = 0.0
            # cm_plot = None
        else:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                action_pred.logits, gt_actions.squeeze().astype(jnp.int32)
            )
            acc = action_pred.action == gt_actions.squeeze()
            # breakpoint_if_cond(acc.mean())
            # jax.debug.breakpoint()

        if self.lam_cfg.model.idm.use_transformer:
            # jax.debug.breakpoint()
            mask = batch.mask[:, :-1]
            loss *= mask
            acc = (acc * mask).sum() / mask.sum()
        else:
            acc = jnp.mean(acc)

        loss = jnp.mean(loss)
        metrics = {"action_loss": loss, "acc": acc, "recon_err": recon_err}
        return loss, (metrics, state)


class LatentActionBaseAgent(BaseAgent):
    def __init__(
        self,
        config,
        key: jax.random.PRNGKey,
        observation_shape: Tuple,
        action_dim: int,
        input_action_dim: int,
        continuous_actions: bool,
        *args,
        **kwargs,
    ):
        super().__init__(
            config,
            key,
            observation_shape,
            config.latent_action_dim,
            config.latent_action_dim,
            continuous_actions=True,
            *args,
            **kwargs,
        )

        # load pretrained IDM/FDM for predicting the latent actions from observations
        config = Path(self.config.lam_ckpt) / "config.json"
        with open(config, "r") as f:
            self.lam_cfg = ConfigDict(json.load(f))

        extra_kwargs = dict(
            observation_shape=observation_shape,
            action_dim=action_dim,
            input_action_dim=input_action_dim,
            continuous_actions=continuous_actions,
        )

        lam_key, decoder_key = jax.random.split(self._init_key, 2)

        # TODO(anthony): hard-coding the checkpoint now, fix this
        self.lam = LatentActionModel(
            self.lam_cfg.model,
            key=lam_key,
            load_from_ckpt=True,
            ckpt_dir=Path(self.config.lam_ckpt),
            **extra_kwargs,
        )

        # load pretrained latent action decoder
        if hasattr(self.config, "latent_action_decoder_ckpt"):
            config_file = Path(self.config.latent_action_decoder_ckpt) / "config.json"
            with open(config_file, "r") as f:
                self.latent_action_decoder_cfg = ConfigDict(json.load(f))

            self.latent_action_decoder = LatentActionDecoder(
                lam=self.lam,
                config=self.latent_action_decoder_cfg.model,
                key=decoder_key,
                load_from_ckpt=True,
                ckpt_dir=Path(self.config.latent_action_decoder_ckpt),
                **extra_kwargs,
            )

    @partial(jax.jit, static_argnames=("self", "is_training"))
    def get_action_jit(self, params, state, rng, env_state, task=None, **kwargs):
        logging.info("inside get action for LatentActionAgent")
        la_key, decoder_key = jax.random.split(rng, 2)

        # if self.config.image_obs:
        #     env_state = einops.rearrange(env_state, "b h w c -> b c h w")

        # predict latent action and then use pretrained action decoder
        la_output, _ = self.model.apply(
            params, state, la_key, self, env_state, task, **kwargs
        )
        latent_actions = la_output.action

        # decode latent action
        action_output, _ = self.latent_action_decoder.model.apply(
            self.latent_action_decoder._params,
            self.latent_action_decoder._state,
            decoder_key,
            self.latent_action_decoder,
            latent_actions=latent_actions,
            is_training=False,
        )
        action_output.latent_action = latent_actions
        return action_output, state


class LatentActionAgent(LatentActionBaseAgent):
    """
    BC Agent that maps observations to latent actions using a pretrained IDM
    """

    @hk.transform_with_state
    def model(self, states, task=None, is_training=True, **kwargs):
        # logging.info(f"action dim: {self.config.latent_action_dim}")
        # predicts the latent action
        policy = ActorCritic(
            self.config.policy,
            is_continuous=self.is_continuous,  # we are predicting latent actions which are continuous vectors
            action_dim=self.config.latent_action_dim,
        )
        policy_output = policy(states, task=task, is_training=is_training)
        return policy_output

    def _init_model(self):
        t, bs = 2, 2
        dummy_state = np.zeros((bs, t, *self.observation_shape), dtype=np.float32)

        if self.config.policy.pass_task_to_policy:
            dummy_task = np.zeros((bs, t, self.task_dim), dtype=np.float32)
        else:
            dummy_task = None

        self._params, self._state = self.model.init(
            self._init_key,
            self,
            states=dummy_state,
            tasks=dummy_task,
            is_training=True,
        )

    def loss_fn(
        self, params: hk.Params, state: hk.State, rng: jax.random.PRNGKey, batch: Tuple
    ):
        logging.info(f"lam loss function, observations: {batch.observations.shape}")

        lam_key, policy_key = jax.random.split(rng, 2)

        # predict the latent action from observations with pretrained model
        lam_output, _ = self.lam.model.apply(
            self.lam._params,
            self.lam._state,
            lam_key,
            self.lam,
            states=batch.observations.astype(jnp.float32),
            is_training=False,
        )
        latent_actions = lam_output.latent_actions

        # if self.config.image_obs:
        #     observations = einops.rearrange(
        #         batch.observations, "b t h w c -> b t c h w"
        #     )
        # else:
        observations = batch.observations

        # only the middle observation matters here
        # not a recurrent policy
        # predict latent action
        action_output, state = self.model.apply(
            params,
            state,
            policy_key,
            self,
            states=observations[:, -2].astype(jnp.float32),
            task=batch.tasks[:, -2],
            is_training=True,
        )
        loss = optax.squared_error(action_output.action, latent_actions)
        loss = jnp.mean(loss)

        metrics = {"bc_loss": loss}
        return loss, (metrics, state)
