import chex
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
    entropy: jnp.ndarray
    dist: Any
    logits: Optional[jnp.ndarray] = None


class ActionHead(hk.Module):
    def __init__(
        self,
        is_continuous: bool,
        action_dim: int,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.01),
        **kwargs
    ):
        super().__init__()
        self.is_continuous = is_continuous
        self.action_dim = action_dim

        init_kwargs = dict(w_init=w_init, b_init=b_init)

        if not is_continuous:
            self.logits = hk.Linear(action_dim, name="discrete_logit", **init_kwargs)
        else:
            self.mean = hk.Linear(action_dim, name="mean", **init_kwargs)
            self.logvar = hk.Linear(action_dim, name="logvar", **init_kwargs)

    def __call__(self, h: jnp.ndarray) -> PolicyOutput:
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
            value=None,
            entropy=action_dist.entropy(),
            dist=action_dist,
            logits=logits,
        )
