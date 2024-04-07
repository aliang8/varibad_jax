import optax
import jax.numpy as jnp


def loss_fn(params, state, ts, batch, rng, continuous_actions=False):
    # trajectory level
    # observations: [B, T, *_]
    observations, actions, rewards = batch

    # [B, T, 1]
    if len(actions.shape) == 2:
        actions = jnp.expand_dims(actions, axis=-1)

    # [B, T]
    mask = jnp.ones_like(rewards)

    # [B, T, 1]
    if len(rewards.shape) == 2:
        rewards = jnp.expand_dims(rewards, axis=-1)

    policy_output, new_state = ts.apply_fn(
        params,
        state,
        rng,
        states=observations.astype(jnp.float32),
        actions=actions,
        rewards=rewards,
        mask=mask,
        is_training=True,
    )

    entropy = policy_output.entropy
    entropy = jnp.mean(entropy)

    action_preds = policy_output.logits

    if continuous_actions:
        # compute MSE loss
        loss = optax.squared_error(action_preds, actions.squeeze(axis=-1))
    else:
        # compute cross entropy with logits
        loss = optax.softmax_cross_entropy_with_integer_labels(
            action_preds, actions.squeeze(axis=-1).astype(jnp.int32)
        )

    loss = jnp.mean(loss)
    metrics = {"bc_loss": loss, "entropy": entropy}

    return loss, (metrics, new_state)
