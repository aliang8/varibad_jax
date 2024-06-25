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
import dm_pix
import flax.linen as nn
import jax.numpy as jnp
from absl import logging
from functools import partial
from pathlib import Path
from ml_collections.config_dict import ConfigDict

from varibad_jax.models.base import BaseModel, BaseAgent
from varibad_jax.agents.actor_critic import ActorCritic, ActorCriticInput
from varibad_jax.models.lam.model import LatentFDM, LatentActionIDM
from varibad_jax.agents.common import ActionHead
from varibad_jax.models.lam.vit_lam import ViTIDMSequence, ViTFDM


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
    stop = x < 0.005
    # stop = x > 0.87

    def true_fn(x):
        jax.debug.breakpoint()

    def false_fn(x):
        pass

    jax.lax.cond(stop, true_fn, false_fn, x)


class LatentActionModel(BaseModel):
    @hk.transform_with_state
    def model(self, states, timestep=None, is_training=True):
        if self.config.use_vit:
            # Use a ViT transformer to encode the images into patches
            idm = ViTIDMSequence(self.config.idm, self.init_kwargs)
            fdm = ViTFDM(self.config.fdm, self.init_kwargs)
        else:
            # LAPO style IDM which takes o_t and o_t+1 as input
            idm = LatentActionIDM(self.config.idm, **self.init_kwargs)
            fdm = LatentFDM(
                self.config.fdm,
                state_dim=self.observation_shape[-1],
                **self.init_kwargs,
            )

        # IDM predicts the latent action (z_t) given o_t-k, ..., o_t and o_t+1
        idm_output = idm(states, timestep=timestep, is_training=is_training)

        if self.config.use_vit:
            latent_actions = idm_output["quantize"][:, :-1]
            timestep = timestep[:, :-1]
        else:
            latent_actions = idm_output["quantize"]

        # FDM predicts o_t+1 given o_t-k, o_t and z_t
        context = states[:, :-1]

        # [B, C, H, W] or [B, T, C, H, W]
        next_state_pred = fdm(
            context,
            # idm_output["quantize"],
            latent_actions,
            timestep=timestep,
            is_training=is_training,
        )

        # if self.config.use_vit:
        #     # we ignore the last prediction since it's the next state
        #     next_state_pred = next_state_pred[:, :-1]

        if self.config.normalize_pred:
            next_state_pred = jnp.tanh(next_state_pred) / 2

        logging.info(f"next_state_pred: {next_state_pred.shape}")

        return LAMOutput(
            next_obs_pred=next_state_pred,
            vq_loss=idm_output["loss"],
            quantize=idm_output["quantize"],
            perplexity=idm_output["perplexity"],
            encoding_indices=idm_output["encoding_indices"],
            # latent_actions=None,
            latent_actions=idm_output["latent_actions"],
        )

    @hk.transform_with_state
    def predict_action(self, latent_actions, is_training):
        x = hk.nets.MLP(
            self.config.mlp_layer_sizes,
            activation=nn.gelu,
            activate_final=True,
            **self.init_kwargs,
        )(latent_actions)
        action_output = ActionHead(
            gaussian_policy=False,
            is_continuous=self.is_continuous,
            action_dim=self.action_dim,
        )(x, is_training)
        return action_output

    @partial(jax.pmap, axis_name="device", static_broadcasted_argnums=(0,))
    def _init_model(self, init_key: jax.random.PRNGKey):
        t, bs = 2 + self.config.context_len, 2
        dummy_states = np.zeros((bs, t, *self.observation_shape), dtype=np.float32)
        # dummy_states = np.zeros((bs, t, 64, 64, 3), dtype=np.float32)
        timesteps = np.arange(t, dtype=np.int32).reshape(1, -1).repeat(bs, axis=0)

        params, state = self.model.init(
            init_key,
            self,
            states=dummy_states,
            timestep=timesteps,
            is_training=True,
        )

        if self.config.add_labelling:
            dummy_latent = np.zeros((bs, t, self.config.idm.code_dim), dtype=np.float32)

            action_head_params, action_head_state = self.predict_action.init(
                init_key,
                self,
                dummy_latent,
                is_training=True,
            )
            params.update(action_head_params)
            state.update(action_head_state)

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

        lam_output, state = self.model.apply(
            params,
            state,
            rng,
            self,
            states=batch.observations.astype(jnp.float32),
            timestep=batch.timestep.astype(jnp.int32),
            is_training=is_training,
        )

        # loss for IDM is VQVAE loss + recon
        # loss for FDM is reconstruction loss
        next_obs_pred = lam_output.next_obs_pred

        if self.config.idm.image_obs:
            # make sure same format as the ground truth, HWC
            if self.config.idm.use_transformer or self.config.use_vit:
                # this should have T-1 predictions where T is our sequence length
                next_obs_pred = einops.rearrange(
                    next_obs_pred, "b t c h w -> b t h w c"
                )
                # gt_next_obs = batch.observations[:, :-1]
                # mask = batch.mask[:, :-1]

                gt_next_obs = batch.observations[:, 1:]
                mask = batch.mask[:, 1:]

                # combine b and t dimensions
                next_obs_pred_bt = einops.rearrange(
                    next_obs_pred, "b t h w c -> (b t) h w c"
                )
                gt_next_obs_bt = einops.rearrange(
                    gt_next_obs, "b t h w c -> (b t) h w c"
                )
                mask_bt = einops.rearrange(mask, "b t -> (b t)")

                mse = dm_pix.mse(next_obs_pred_bt, gt_next_obs_bt)
                mse *= mask_bt
                mse = jnp.sum(mse) / jnp.sum(mask)

                ssim = dm_pix.ssim(next_obs_pred_bt, gt_next_obs_bt)
                ssim *= mask_bt
                ssim = jnp.sum(ssim) / jnp.sum(mask)

                psnr = None

                recon_loss = optax.squared_error(next_obs_pred, gt_next_obs)
                recon_loss = jnp.mean(recon_loss, axis=(2, 3, 4))
                recon_loss *= mask
                recon_loss = jnp.sum(recon_loss) / jnp.sum(mask)
                # recon_loss = (1 - ssim) + mse
                # recon_loss = mse
            else:
                next_obs_pred = einops.rearrange(next_obs_pred, "b c h w -> b h w c")
                gt_next_obs = batch.observations[:, -1]

                # here we are using MSE loss for computing image reconstruction loss
                # we could try other losses like SSIM, perceptual loss, etc.
                # or a combination of losses

                ssim = dm_pix.ssim(next_obs_pred, gt_next_obs).mean()
                mse = dm_pix.mse(next_obs_pred, gt_next_obs).mean()
                psnr = dm_pix.psnr(next_obs_pred, gt_next_obs).mean()

                recon_loss = (1 - ssim) + mse
                # recon_loss = optax.squared_error(next_obs_pred, gt_next_obs)
                # recon_loss = jnp.mean(recon_loss)
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

        if self.config.idm.image_obs:
            metrics.update({"ssim": ssim, "mse": mse, "psnr": psnr})

        loss = self.config.beta_loss_weight * lam_output.vq_loss + recon_loss

        # if batch.labelled is not None:
        #     jax.debug.breakpoint()
        #     # if the transition is labelled, we can compute action decoding loss
        #     latent_actions = lam_output.latent_actions
        #     action_pred, action_state = self.predict_action.apply(
        #         params,
        #         state,
        #         rng,
        #         self,
        #         latent_actions=latent_actions,
        #         is_training=is_training,
        #     )

        #     gt_actions = batch.actions[:, -2]

        #     if gt_actions.shape[-1] == 1:
        #         gt_actions = gt_actions.squeeze(axis=-1)

        #     action_pred_loss = optax.softmax_cross_entropy_with_integer_labels(
        #         action_pred.logits, gt_actions.astype(jnp.int32)
        #     )
        #     mask = batch.labelled[:, -2]
        #     action_pred_loss *= mask
        #     action_pred_loss = action_pred_loss.sum() / mask.sum()

        #     acc = action_pred.action == gt_actions
        #     acc *= mask
        #     acc = acc.sum() / mask.sum()

        #     metrics.update({"action_pred_loss": action_pred_loss, "acc": acc})
        #     state.update(action_state)
        #     loss += action_pred_loss
        # else:
        #     metrics.update({"action_pred_loss": 0.0, "acc": 0.0})

        extra = {"next_obs_pred": next_obs_pred}

        return loss, (metrics, extra, state)


class LatentActionDecoder(BaseModel):
    """
    Model that decodes latent actions to ground truth actions. Use in conjunction with the
    IDM model learned with LAPO.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #     if lam is None:
        #         # load pretrained latent action model
        config = Path(self.config.lam_ckpt) / "config.json"
        with open(config, "r") as f:
            self.lam_cfg = ConfigDict(json.load(f))

    #         self.lam = LatentActionModel(
    #             self.lam_cfg.model,  # TODO: fix me
    #             key=self._init_key,
    #             observation_shape=self.observation_shape,
    #             action_dim=self.action_dim,
    #             input_action_dim=self.input_action_dim,
    #             continuous_actions=self.is_continuous,
    #             load_from_ckpt=True,
    #             # ckpt_file=Path(self.config.lam_ckpt) / "model_ckpts" / "ckpt_0014.pkl",
    #             ckpt_dir=Path(self.config.lam_ckpt),
    #         )
    #     else:
    #         self.lam = lam

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

    @partial(jax.pmap, axis_name="device", static_broadcasted_argnums=(0,))
    def _init_model(self, init_key: jax.random.PRNGKey):
        bs = 2
        dummy_latent_actions = np.zeros(
            (bs, self.config.latent_action_dim), dtype=np.float32
        )

        params, state = self.model.init(
            init_key,
            self,
            latent_actions=dummy_latent_actions,
            is_training=True,
        )
        return params, state

    def loss_fn(
        self,
        params: hk.Params,
        state: hk.State,
        rng: jax.random.PRNGKey,
        batch,
        is_training: bool,
    ):
        # observations = batch.observations

        # agent_key, decoder_key = jax.random.split(rng, 2)

        # model_output, _ = self.lam.model.apply(
        #     self.lam._ts.params,
        #     self.lam._ts.state,
        #     agent_key,
        #     self.lam,
        #     states=observations.astype(jnp.float32),
        #     is_training=False,
        # )

        # jax.debug.breakpoint()

        # obs_pred = model_output.next_obs_pred
        # if self.lam_cfg.model.idm.image_obs:
        #     obs_pred = einops.rearrange(obs_pred, "b c h w -> b h w c")

        # gt_next_obs = observations[:, -1]
        # recon_err = optax.squared_error(obs_pred, gt_next_obs)
        # recon_err = jnp.mean(recon_err)

        # latent_actions = model_output.latent_actions

        rng, decoder_key = jax.random.split(rng, 2)
        # latent_actions = batch.actions[:, -2]
        # latent_actions = batch.latent_actions[:, -2]
        latent_actions = batch.latent_actions

        # jax.debug.breakpoint()

        # decode latent action to ground truth action
        action_pred, state = self.model.apply(
            params,
            state,
            decoder_key,
            self,
            latent_actions=latent_actions,
            is_training=is_training,
        )

        # if self.lam_cfg.model.idm.use_transformer:
        #     gt_actions = batch.actions[:, :-1]
        # else:
        #     # TODO: make sure i'm predicting the right action
        #     gt_actions = batch.actions[:, -2]
        gt_actions = batch.actions

        if self.is_continuous:
            loss = optax.squared_error(action_pred.action, gt_actions)
            acc = 0.0
            # cm_plot = None
        else:
            # logging.info(
            #     f"action logits: {action_pred.logits.shape}, gt_actions: {gt_actions.shape}"
            # )
            if gt_actions.shape[-1] == 1:
                gt_actions = gt_actions.squeeze(axis=-1)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                action_pred.logits, gt_actions.astype(jnp.int32)
            )
            acc = action_pred.action == gt_actions
            # breakpoint_if_cond(acc.mean())

        if self.lam_cfg.model.idm.use_transformer:
            # jax.debug.breakpoint()
            mask = batch.mask[:, :-1]
            loss *= mask
            acc = (acc * mask).sum() / mask.sum()
        else:
            acc = jnp.mean(acc)

        loss = jnp.mean(loss)
        # metrics = {"action_loss": loss, "acc": acc, "recon_err": recon_err}
        metrics = {"action_loss": loss, "acc": acc}
        extras = {}
        return loss, (metrics, extras, state)


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
        # config = Path(self.config.lam_ckpt) / "config.json"
        # with open(config, "r") as f:
        #     self.lam_cfg = ConfigDict(json.load(f))

        extra_kwargs = dict(
            observation_shape=observation_shape,
            action_dim=action_dim,
            input_action_dim=input_action_dim,
            continuous_actions=continuous_actions,
        )

        # lam_key, decoder_key = jax.random.split(self._init_key, 2)

        # self.lam = LatentActionModel(
        #     self.lam_cfg.model,
        #     key=lam_key,
        #     load_from_ckpt=True,
        #     # ckpt_file=Path(self.config.lam_ckpt) / "model_ckpts" / "ckpt_0100.pkl",
        #     ckpt_dir=Path(self.config.lam_ckpt),
        #     **extra_kwargs,
        # )

        rng, decoder_key = jax.random.split(self._init_key[0], 2)

        # load pretrained latent action decoder
        if hasattr(self.config, "latent_action_decoder_ckpt"):
            ckpt_path = self.config.latent_action_decoder_ckpt
            if "idm_nt" in self.config and self.config.idm_nt != -1:
                ckpt_path = re.sub(r"nt-\d+", f"nt-{self.config.idm_nt}", ckpt_path)

            config_file = Path(ckpt_path) / "config.json"
            with open(config_file, "r") as f:
                self.latent_action_decoder_cfg = ConfigDict(json.load(f))

            self.latent_action_decoder = LatentActionDecoder(
                # lam=self.lam,
                config=self.latent_action_decoder_cfg.model,
                key=decoder_key,
                load_from_ckpt=True,
                ckpt_dir=Path(ckpt_path),
                **extra_kwargs,
            )

            self.decode = self.latent_action_decoder.model.apply
            self.la_decoder_params = jax.tree_map(
                lambda x: x[0], self.latent_action_decoder._ts.params
            )
            self.la_decoder_state = jax.tree_map(
                lambda x: x[0], self.latent_action_decoder._ts.state
            )

    @partial(jax.jit, static_argnames=("self", "is_training"))
    def get_action_jit(self, ts, rng, env_state, task=None, **kwargs):
        logging.info("inside get action for LatentActionAgent")
        la_key, decoder_key = jax.random.split(rng, 2)

        params = jax.tree_util.tree_map(lambda x: x[0], ts.params)
        state = jax.tree_util.tree_map(lambda x: x[0], ts.state)

        # if self.config.image_obs:
        #     env_state = einops.rearrange(env_state, "b h w c -> b c h w")

        # predict latent action and then use pretrained action decoder
        (la_output, hstate), state = self.model.apply(
            params, state, la_key, self, env_state, task, **kwargs
        )
        latent_actions = la_output.action

        # decode latent action
        action_output, _ = self.decode(
            self.la_decoder_params,
            self.la_decoder_state,
            decoder_key,
            self.latent_action_decoder,
            latent_actions=latent_actions,
            is_training=False,
        )
        action_output.latent_action = latent_actions
        return (action_output, hstate), state


class LatentActionAgent(LatentActionBaseAgent):
    """
    BC Agent that maps observations to latent actions using a pretrained IDM
    """

    @hk.transform_with_state
    def model(self, states, task=None, is_training=True, **kwargs):
        logging.info(f"action dim: {self.config.latent_action_dim}")
        logging.info(f"states shape: {states.shape}")

        # predicts the latent action
        policy = ActorCritic(
            self.config.policy,
            is_continuous=self.is_continuous,  # we are predicting latent actions which are continuous vectors
            action_dim=self.config.latent_action_dim,
        )
        policy_input = ActorCriticInput(
            state=states, task=task, is_training=is_training
        )
        policy_output = policy(policy_input, hidden_state=None)
        return policy_output

    @partial(jax.pmap, axis_name="device", static_broadcasted_argnums=(0,))
    def _init_model(self, init_key: jax.random.PRNGKey):
        bs = 2
        dummy_state = np.zeros((bs, *self.observation_shape), dtype=np.float32)

        if self.config.policy.pass_task_to_policy:
            dummy_task = np.zeros((bs, self.task_dim), dtype=np.float32)
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

        # lam_key, policy_key, decoder_key = jax.random.split(rng, 3)

        # predict the latent action from observations with pretrained model
        # lam_output, _ = self.lam.model.apply(
        #     self.lam._ts.params,
        #     self.lam._ts.state,
        #     lam_key,
        #     self.lam,
        #     states=batch.observations.astype(jnp.float32),
        #     is_training=False,
        # )
        # latent_actions = lam_output.latent_actions

        # if self.config.image_obs:
        #     observations = einops.rearrange(
        #         batch.observations, "b t h w c -> b t c h w"
        #     )
        # else:
        rng, policy_key, decoder_key = jax.random.split(rng, 3)
        observations = batch.observations
        latent_actions = batch.latent_actions

        # only the middle observation matters here
        # not a recurrent policy
        # predict latent action
        (action_output, _), state = self.model.apply(
            params,
            state,
            policy_key,
            self,
            states=observations.astype(jnp.float32),
            task=batch.tasks if batch.tasks is not None else None,
            is_training=is_training,
        )
        loss = optax.squared_error(action_output.action, latent_actions)
        loss = jnp.mean(loss)

        # decode latent action and record prediction acc
        # this measures how close the LA + decoder is to the gt action
        # output from LAM => action decoder
        gt_action_output, _ = self.decode(
            self.la_decoder_params,
            self.la_decoder_state,
            decoder_key,
            self.latent_action_decoder,
            latent_actions=latent_actions,
            is_training=False,
        )

        gt_actions = batch.actions
        acc = gt_action_output.action == gt_actions.squeeze()
        acc = jnp.mean(acc)

        # this measures how close the LA pred + decoder is to the gt action
        # output from policy => action decoder
        decoded_action_output, _ = self.decode(
            self.la_decoder_params,
            self.la_decoder_state,
            decoder_key,
            self.latent_action_decoder,
            latent_actions=action_output.action,
            is_training=False,
        )
        decoded_acc = decoded_action_output.action == gt_actions.squeeze()
        decoded_acc = jnp.mean(decoded_acc)

        metrics = {"bc_loss": loss, "acc": acc, "decoded_acc": decoded_acc}

        extras = {}
        return loss, (metrics, extras, state)
