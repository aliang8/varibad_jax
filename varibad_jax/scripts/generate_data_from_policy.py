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
from varibad_jax.envs.xland_utils import make_envs
from varibad_jax.utils.rollout import eval_rollout
from varibad_jax.models.varibad.helpers import encode_trajectory, decode, get_prior

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

    config_p = FrozenConfigDict(config_p)

    env = make_envs(
        env_id=config.env.env_id,
        seed=config.seed,
        num_envs=config.env.num_processes,
        num_episodes_per_rollout=config.env.num_episodes_per_rollout,
        benchmark_path=config.env.benchmark_path,
        ruleset_id=config.env.ruleset_id,
        training=True,
    )
    action_dim = env.action_space.n
    continuous_actions = not isinstance(env.action_space, gym.spaces.Discrete)
    if continuous_actions:
        vae_action_dim = action_dim
    else:
        vae_action_dim = 1

    # create train states
    ts_vae, ts_policy = create_ts(
        config, next(rng_seq), env, vae_action_dim, action_dim
    )

    get_prior_fn = functools.partial(
        jax.jit(
            get_prior.apply,
            static_argnames=("config", "batch_size"),
        ),
        config=FrozenConfigDict(config.vae),
    )

    # collect some rollouts
    steps_per_rollout = config.env.num_episodes_per_rollout * env.max_episode_steps

    rng_keys = jax.random.split(next(rng_seq), config.num_rollouts_collect)

    rollout_env = make_envs(
        env_id=config.env.env_id,
        seed=config.seed,
        num_envs=config.env.num_processes,
        num_episodes_per_rollout=config.env.num_episodes_per_rollout,
        benchmark_path=config.env.benchmark_path,
        ruleset_id=config.env.ruleset_id,
        training=False,
    )

    logging.info("start data collection")
    stats, transitions = jax.vmap(
        eval_rollout,
        in_axes=(0, None, None, None, None, None, None, None),
    )(
        rng_keys,
        rollout_env,
        config,
        ts_policy,
        ts_vae,
        get_prior_fn,
        vae_action_dim,
        steps_per_rollout,
    )

    metrics = {
        "return": jnp.mean(stats["reward"]),
        "avg_length": jnp.mean(stats["length"]),
    }
    logging.info(f"eval metrics: {metrics}")

    # save transitions to a dataset format
    data_dir = Path(config.root_dir) / "datasets"
    data_dir.mkdir(exist_ok=True, parents=True)
    data_file = (
        data_dir
        / f"eid-{config.env.env_id}_n-{config.num_rollouts_collect}_steps-{steps_per_rollout}.pkl"
    )
    with open(data_file, "wb") as f:
        pickle.dump(transitions, f)


if __name__ == "__main__":
    app.run(main)
