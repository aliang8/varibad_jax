from typing import Any, Optional, Tuple

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

tfd = tfp.distributions
tfb = tfp.bijectors


@chex.dataclass
class PolicyOutput:
    action: jnp.ndarray
    value: jnp.ndarray
    entropy: jnp.ndarray
    dist: Any
    logits: Optional[jnp.ndarray] = None


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
        self.state_embed = hk.Linear(
            self.config.embedding_dim, name="state_embed", **init_kwargs
        )
        # self.state_embed = hk.Embed(vocab_size=25, embed_dim=self.config.embedding_dim)
        self.action_embed = hk.Linear(
            self.config.embedding_dim, name="action_embed", **init_kwargs
        )

        if self.config.pass_latent_to_policy:
            self.latent_embed = hk.Linear(
                self.config.embedding_dim, name="latent_embed", **init_kwargs
            )

        if self.config.pass_task_to_policy:
            self.task_embed = hk.Linear(
                self.config.embedding_dim, name="task_embed", **init_kwargs
            )

        self.critic_mlp = hk.nets.MLP(
            list(self.config.mlp_layers) + [1],
            activation=nn.gelu,
            activate_final=False,
            name="critic",
            **init_kwargs,
        )

        self.action_pred = hk.nets.MLP(
            list(self.config.mlp_layers),
            activation=nn.gelu,
            activate_final=True,
            name="action_pred",
            **init_kwargs,
        )

        self.is_continuous = is_continuous

        if not is_continuous:
            self.logits = hk.Linear(action_dim, name="discrete_logit", **init_kwargs)
        else:
            self.mean = hk.Linear(action_dim, name="mean", **init_kwargs)
            self.logvar = hk.Linear(action_dim, name="logvar", **init_kwargs)

    def __call__(
        self, state: jnp.ndarray, latent: jnp.ndarray = None, task: jnp.ndarray = None
    ):
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

        value = self.critic_mlp(policy_input)

        h = self.action_pred(policy_input)

        if not self.is_continuous:
            logits = self.logits(h)
            action_dist = tfd.Categorical(logits=logits)
            action = action_dist.sample(seed=hk.next_rng_key())
        else:
            mean = self.mean(h)
            logvar = self.logvar(h)
            action_dist = tfd.MultivariateNormalDiag(
                loc=mean, scale_diag=jnp.exp(logvar)
            )
            action = action_dist.sample(seed=hk.next_rng_key())

        return PolicyOutput(
            action=action,
            value=value,
            entropy=action_dist.entropy(),
            dist=action_dist,
            logits=logits,
        )
