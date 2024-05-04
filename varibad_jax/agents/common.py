import jax
import chex
import optax
import haiku as hk
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from typing import Optional, Any

tfd = tfp.distributions
tfb = tfp.bijectors


@chex.dataclass
class PolicyOutput:
    action: jnp.ndarray
    value: jnp.ndarray
    entropy: Optional[jnp.ndarray] = None
    dist: Optional[tfd.Distribution] = None
    logits: Optional[jnp.ndarray] = None
    latent_action: Optional[jnp.ndarray] = None


class ActionHead(hk.Module):
    def __init__(
        self,
        gaussian_policy: bool,
        is_continuous: bool,
        action_dim: int,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.01),
        **kwargs
    ):
        super().__init__()
        self.is_continuous = is_continuous
        self.action_dim = action_dim
        self.gaussian_policy = gaussian_policy

        init_kwargs = dict(w_init=w_init, b_init=b_init)

        if not is_continuous:
            self.logits = hk.Linear(action_dim, name="discrete_logit", **init_kwargs)
        else:
            self.mean = hk.Linear(action_dim, name="mean", **init_kwargs)
            self.logvar = hk.Linear(action_dim, name="logvar", **init_kwargs)

    def __call__(self, h: jnp.ndarray, is_training: bool = True) -> PolicyOutput:
        if not self.is_continuous:
            logits = self.logits(h)
            action_dist = tfd.Categorical(logits=logits)
            if is_training:
                action = action_dist.sample(seed=hk.next_rng_key())
            else:
                action = action_dist.mode()
            entropy = action_dist.entropy()

        else:
            mean = self.mean(h)
            logvar = self.logvar(h)
            if self.gaussian_policy:
                action_dist = tfd.MultivariateNormalDiag(
                    loc=mean, scale_diag=jnp.exp(logvar)
                )
                action = action_dist.sample(seed=hk.next_rng_key())
                logits = jnp.stack([mean, logvar], axis=-1)
                entropy = action_dist.entropy()
            else:
                action = mean
                logits = None
                entropy = None
                action_dist = None

        return PolicyOutput(
            action=action,
            value=None,
            entropy=entropy,
            dist=action_dist,
            logits=logits,
        )
