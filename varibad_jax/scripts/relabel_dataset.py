"""
script for collecting offline dataset from trained policy
currently supports only LSTM varibad policy
"""

from absl import app, logging
import einops
import re
import os
from pathlib import Path
import pickle
from PIL import Image
import numpy as np
import jax
import tqdm
import json
import jax.numpy as jnp
import functools
import haiku as hk
import gymnax
import gymnasium as gym
import jax.tree_util as jtu
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags

from varibad_jax.envs.utils import make_envs, make_procgen_envs
from varibad_jax.utils.rollout import run_rollouts
from varibad_jax.models.varibad.varibad import VariBADModel
from varibad_jax.agents.ppo.ppo import PPOAgent
from varibad_jax.models.base import RandomActionAgent
from varibad_jax.utils.data_utils import (
    merge_trajectories,
    split_data_into_trajectories,
)
from varibad_jax.models.lam.lam import LatentActionModel, LatentActionDecoder

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_CONFIG = config_flags.DEFINE_config_file("config")


def main(_):
    config = _CONFIG.value

    # load pretrained IDM/FDM for predicting the latent actions from observations
    logging.info("loading LAM")
    lam_cfg_path = Path(config.model.lam_ckpt) / "config.json"
    with open(lam_cfg_path, "r") as f:
        lam_cfg = ConfigDict(json.load(f))

    lam_cfg = ConfigDict(lam_cfg)
    envs = make_procgen_envs(num_envs=1, env_id=lam_cfg.env.env_id, gamma=1)
    # continuous_actions = not isinstance(
    #     envs.action_space, gym.spaces.Discrete
    # ) and not isinstance(envs.action_space, gymnax.environments.spaces.Discrete)
    # if continuous_actions:
    #     input_action_dim = action_dim = envs.action_space.shape[0]
    # else:
    continuous_actions = False
    action_dim = envs.action_space.n
    input_action_dim = 1

    rng_seq = hk.PRNGSequence(lam_cfg.seed)
    extra_kwargs = dict(
        observation_shape=envs.single_observation_space.shape,
        action_dim=action_dim,
        input_action_dim=input_action_dim,
        continuous_actions=continuous_actions,
    )

    lam = LatentActionModel(
        lam_cfg.model,
        key=next(rng_seq),
        load_from_ckpt=True,
        # ckpt_file=Path(config.lam_ckpt) / "model_ckpts" / "ckpt_0100.pkl",
        ckpt_dir=Path(config.model.lam_ckpt),
        **extra_kwargs,
    )
    logging.info("done loading LAM")

    # load original dataset

    logging.info("loading original dataset")

    # load chunks
    training_env_id = config.env.env_id

    data_dir = Path(config.root_dir) / config.data.data_dir

    for split in ["train", "test"]:
        chunk_dir = (
            data_dir
            / Path("procgen")
            / "expert_data"
            / training_env_id
            / split
            / "chunks"
        )
        # create a new directory for relabelled dataset
        relabel_dir = (
            data_dir
            / Path("procgen")
            / "expert_data"
            / training_env_id
            / split
            / "la_chunks"
        )
        relabel_dir.mkdir(exist_ok=True, parents=True)

        chunk_files = list(chunk_dir.glob("*.npz"))
        logging.info("number of chunks: %d", len(chunk_files))

        jit_apply = jax.jit(lam.model.apply, static_argnames=("self", "is_training"))

        for i, chunk_file in tqdm.tqdm(
            enumerate(chunk_files), desc=f"relabelling {split} files"
        ):
            data = np.load(chunk_file)

            # replace actions with latent actions
            observations = data["observations"]

            # make it into LAPO style data, since we're loading in trajectories
            latent_actions = []
            window_size = 2 + lam_cfg.model.context_len

            for j in tqdm.tqdm(range(observations.shape[1] - window_size + 1)):
                window = observations[:, j : j + window_size]
                lam_output, _ = jit_apply(
                    lam._ts.params,
                    lam._ts.state,
                    next(rng_seq),
                    lam,
                    states=window.astype(jnp.float32),
                    is_training=False,
                )
                latent_action = lam_output.quantize
                latent_actions.append(latent_action)

            latent_actions = np.stack(latent_actions, axis=1)
            b, _, d = latent_actions.shape
            # add zeros at the end
            zeros = np.zeros((b, 1, d))
            pre = np.zeros((b, lam_cfg.model.context_len, d))
            latent_actions = np.concatenate([pre, latent_actions, zeros], axis=1)

            # latent_actions = einops.rearrange(latent_actions, "b t ... -> (b t) ...")

            new_data = dict(data)
            new_data["latent_actions"] = latent_actions
            relabel_file = relabel_dir / f"chunk_{i}.npz"

            with open(relabel_file, "wb") as f:
                np.savez_compressed(f, **new_data)


if __name__ == "__main__":
    app.run(main)
