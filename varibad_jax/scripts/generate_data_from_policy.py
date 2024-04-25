"""
script for collecting offline dataset from trained policy
currently supports only LSTM varibad policy
"""

from absl import app, logging
import os
from pathlib import Path
import pickle
import jax
import json
import jax.numpy as jnp
import functools
import haiku as hk
import gymnasium as gym
import jax.tree_util as jtu
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

    model_ckpt_dir = Path(config.root_dir) / config.model_ckpt_dir / "model_ckpts"
    ckpt_data = model_ckpt_dir / "best.txt"
    with open(ckpt_data, "r") as f:
        ckpt_data = f.read()
        print(ckpt_data)

    ckpt_config_file = Path(config.root_dir) / config.model_ckpt_dir / "config.json"
    with open(ckpt_config_file, "r") as f:
        config_p = json.load(f)

    config_p = ConfigDict(config_p)
    config_p.load_from_ckpt = True
    config_p.ckpt_step = config.ckpt_step
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
    data_dir = Path(config.root_dir) / "datasets" / f"eid-{config.env.env_id}_n-{config.num_rollouts_collect}_steps-{steps_per_rollout}"
    data_dir.mkdir(exist_ok=True, parents=True)
    data_file = data_dir / "dataset.pkl"
    metadata_file = data_dir / "metadata.json"

    # make into list of pytrees
    @jax.jit
    def get_ts(i):
        return jtu.tree_map(lambda y: y[i], transitions)

    dataset_size = actions.shape[0]
    observations = jnp.array([get_ts(i).observation for i in range(dataset_size)])
    rewards = jnp.array([get_ts(i).reward for i in range(dataset_size)])
    dones = jnp.array([get_ts(i).last() for i in range(dataset_size)])

    # actions - [T, B, 1]
    # observations - [T, B, ...]
    data = dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )

    metadata = {
        "pretrained_model_config": config_p.to_dict(),
        "ckpt_data": ckpt_data,
        "ckpt_dir": str(model_ckpt_dir),
        "avg_ep_return": float(eval_metrics["episode_return"])
    }
    with open(data_file, "wb") as f:
        pickle.dump(data, f)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    app.run(main)
