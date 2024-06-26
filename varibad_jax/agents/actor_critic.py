from typing import Any, Optional, Tuple

import einops
import chex
import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from ml_collections.config_dict import ConfigDict
import numpy as np
from tensorflow_probability.substrates import jax as tfp
import gymnasium as gym
from varibad_jax.models.common import ImageEncoder
from varibad_jax.agents.common import ActionHead

tfd = tfp.distributions
tfb = tfp.bijectors


@chex.dataclass
class ActorCriticInput:
    state: jnp.ndarray
    is_training: bool = None
    latent: Optional[jnp.ndarray] = None
    task: Optional[jnp.ndarray] = None


class ActorCritic(hk.Module):
    """Critic+Actor for PPO."""

    def __init__(
        self,
        config: ConfigDict,
        is_continuous: bool,
        action_dim: int,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.01),
    ):
        super().__init__(name="ActorCritic")
        init_kwargs = dict(w_init=w_init, b_init=b_init)

        self.config = config
        self.image_obs = config.image_obs
        if self.image_obs:
            self.state_embed = ImageEncoder(
                **config.image_encoder_config, **init_kwargs
            )
        else:
            self.state_embed = hk.Linear(
                self.config.embedding_dim, name="state_embed", **init_kwargs
            )
        # self.state_embed = hk.Embed(vocab_size=25, embed_dim=self.config.embedding_dim)
        # self.action_embed = hk.Linear(
        #     self.config.embedding_dim, name="action_embed", **init_kwargs
        # )

        if self.config.pass_latent_to_policy:
            self.latent_embed = hk.Linear(
                self.config.embedding_dim, name="latent_embed", **init_kwargs
            )

        if self.config.pass_task_to_policy:
            self.task_embed = hk.Linear(
                self.config.embedding_dim, name="task_embed", **init_kwargs
            )

        self.critic_mlp = hk.nets.MLP(
            list(self.config.mlp_layer_sizes) + [1],
            activation=nn.gelu,
            activate_final=False,
            name="critic",
            **init_kwargs,
        )

        if self.config.use_rnn_policy:
            self.rnn = hk.GRU(
                hidden_size=self.config.rnn_hidden_size,
                name="rnn",
                w_i_init=w_init,
                w_h_init=w_init,
                b_init=b_init,
            )

        self.embed_input = hk.nets.MLP(
            list(self.config.mlp_layer_sizes),
            activation=nn.gelu,
            activate_final=True,
            name="embed_input",
            **init_kwargs,
        )

        self.action_head = ActionHead(
            gaussian_policy=self.config.gaussian_policy,
            is_continuous=is_continuous,
            action_dim=action_dim,
            **init_kwargs,
        )

    def initial_state(self, batch_size: int):
        return self.rnn.initial_state(batch_size)

    def __call__(
        self,
        policy_input: ActorCriticInput,
        hidden_state: Optional[jnp.ndarray] = None,
    ):
        is_training = (
            policy_input.is_training if policy_input.is_training is not None else True
        )
        state = policy_input.state
        latent = policy_input.latent
        task = policy_input.task

        if self.image_obs:
            state = einops.rearrange(state, "... h w c -> ... c h w")
            state_embed = self.state_embed(state, is_training=is_training)
        else:
            state_embed = self.state_embed(state)

        state_embed = nn.gelu(state_embed)
        policy_input = state_embed

        # Encode latent
        if self.config.pass_latent_to_policy and latent is not None:
            latent_embed = self.latent_embed(latent)
            latent_embed = nn.gelu(latent_embed)
            policy_input = jnp.concatenate((policy_input, latent_embed), axis=-1)

        # Encode task
        if self.config.pass_task_to_policy and task is not None:
            task_embed = self.task_embed(task)
            task_embed = nn.gelu(task_embed)
            policy_input = jnp.concatenate((policy_input, task_embed), axis=-1)

        h = self.embed_input(policy_input)

        if self.config.use_rnn_policy:
            if hidden_state is None:
                hidden_state = self.rnn.initial_state(state.shape[0])
            h, hidden_state = self.rnn(h, hidden_state)
        else:
            hidden_state = None

        value = self.critic_mlp(h)
        policy_output = self.action_head(h, is_training=is_training)
        policy_output.value = value
        return policy_output, hidden_state
