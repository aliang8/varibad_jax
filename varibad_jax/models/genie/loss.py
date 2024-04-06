import jax
import optax
import haiku as hk
import jax.numpy as jnp
from flax.training.train_state import TrainState
from ml_collections.config_dict import FrozenConfigDict


def loss_fn(params, ts: TrainState, state: hk.State, batch, rng: jax.random.PRNGKey):
    observations, actions, rewards = batch

    # observation - [B, T, H, W, C]
    lam_output, new_state = ts.apply_fn(
        params, state, rng, obs=observations.astype(jnp.float32)
    )

    # [B, T-1, H, W, C]
    next_obs_pred = lam_output.obs_pred[:, 1:]
    next_obs_gt = observations[:, 1:].astype(jnp.float32)

    # for image reconstruction, MSE or MAE
    recon_loss = optax.squared_error(next_obs_gt, next_obs_pred)
    recon_loss = recon_loss.mean(axis=0).sum(axis=0).mean()

    loss = recon_loss + lam_output.codebook_loss
    loss = jnp.mean(loss)
    metrics = {
        "recon_loss": recon_loss,
        "codebook_loss": lam_output.codebook_loss,
    }
    return loss, (metrics, state)
