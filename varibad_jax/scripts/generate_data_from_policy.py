"""
script for collecting offline dataset from trained policy
currently supports only LSTM varibad policy
"""

from absl import app, logging
import os
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
import functools
import haiku as hk
import gymnasium as gym
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from varibad_jax.trainers.meta_trainer import create_ts
from varibad_jax.envs.utils import make_envs
from varibad_jax.utils.rollout import run_rollouts
from varibad_jax.models.varibad.helpers import get_prior

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_CONFIG = config_flags.DEFINE_config_file("config")


def main(_):
    config = _CONFIG.value
    # first load models from checkpoint
    rng_seq = hk.PRNGSequence(config.seed)

    model_ckpt_dir = Path(config.root_dir) / config.model_ckpt_dir
    ckpt_file = model_ckpt_dir / f"ckpt_{config.checkpoint_step}.pkl"

    logging.info(f"loading models from {ckpt_file}")
    with open(ckpt_file, "rb") as f:
        ckpt = pickle.load(f)
        config_p = ckpt["config"]

    config_p = ConfigDict(config_p)
    config_p.load_from_ckpt = True
    config_p.checkpoint_step = config.checkpoint_step
    config_p.model_ckpt_dir = config.model_ckpt_dir
    print(config_p)

    envs, env_params = make_envs(**config.env, training=False)
    continuous_actions = not isinstance(envs.action_space, gym.spaces.Discrete)

    if continuous_actions:
        input_action_dim = action_dim = envs.action_space.shape[0]
    else:
        action_dim = envs.action_space.n
        input_action_dim = 1

    # create train states
    ts_vae, ts_policy, vae_state, policy_state = create_ts(
        config_p, next(rng_seq), envs, input_action_dim, action_dim, num_update_steps=0
    )

    get_prior_fn = functools.partial(
        jax.jit(
            get_prior.apply,
            static_argnames=("config", "batch_size"),
        ),
        config=FrozenConfigDict(config_p.vae),
    )

    # collect some rollouts
    steps_per_rollout = config.env.num_episodes_per_rollout * envs.max_episode_steps

    logging.info("start data collection")
    config_p.num_eval_rollouts = config.num_rollouts_collect
    eval_metrics, transitions, actions = run_rollouts(
        rng=next(rng_seq),
        state=[vae_state, policy_state],
        env=envs,
        config=config_p,
        ts_policy=ts_policy,
        ts_vae=ts_vae,
        get_prior=get_prior_fn,
        steps_per_rollout=steps_per_rollout,
        action_dim=input_action_dim,
    )

    logging.info(f"eval metrics: {eval_metrics}")

    # save transitions to a dataset format
    data_dir = Path(config.root_dir) / "datasets"
    data_dir.mkdir(exist_ok=True, parents=True)
    data_file = (
        data_dir
        / f"eid-{config.env.env_id}_n-{config.num_rollouts_collect}_steps-{steps_per_rollout}.pkl"
    )
    data = {
        "config": config_p.to_dict(),
        "ckpt_file": ckpt_file,
        "transitions": transitions,
        "actions": actions,
    }
    with open(data_file, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    app.run(main)
