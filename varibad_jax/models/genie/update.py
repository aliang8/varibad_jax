import jax
import optax
from flax.training.train_state import TrainState
from ml_collections.config_dict import FrozenConfigDict


def update(
    params, ts: TrainState, rng: jax.random.PRNGKey, config: FrozenConfigDict, batch
):
    # observation - [B, T, H, W, C]
    lam_output = ts.apply_fn(params, rng, batch.obs)

    # [B, 1, H, W, C]
    next_obs_pred = lam_output.obs_pred

    next_obs_gt = batch.obs[:, 1:]

    # for image reconstruction, MSE or MAE
    recon_loss = optax.squared_loss(next_obs_gt, next_obs_pred).mean()

    loss = recon_loss + lam_output.codebook_loss

    metrics = {
        "recon_loss": recon_loss,
        "codebook_loss": lam_output.codebook_loss,
    }
    return loss, metrics
