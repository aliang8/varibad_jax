"""
script for relabelling offline dataset 
with actions predicted from vpt idm
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
from varibad_jax.models.vpt.vpt import VPT

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_CONFIG = config_flags.DEFINE_config_file("config")

def main(_):
    config = _CONFIG.value

    # load pretrained VPT IDM for predicting the actions from observations
    logging.info("loading IDM")
    
    vpt_idm_cfg_path = Path(config.model.vpt_idm_ckpt) / "config.json"
    with open(vpt_idm_cfg_path, "r") as f:
        vpt_idm_cfg = ConfigDict(json.load(f))

    vpt_idm_cfg = ConfigDict(vpt_idm_cfg)
    #envs = make_procgen_envs(num_envs=1, env_id=lam_cfg.env.env_id, gamma=1)
    envs, _ = make_envs(**vpt_idm_cfg.env, training=False)
    # continuous_actions = not isinstance(
    #     envs.action_space, gym.spaces.Discrete
    # ) and not isinstance(envs.action_space, gymnax.environments.spaces.Discrete)
    # if continuous_actions:
    #     input_action_dim = action_dim = envs.action_space.shape[0]
    # else:
    continuous_actions = False
    action_dim = envs.action_space.n
    input_action_dim = 1

    rng_seq = hk.PRNGSequence(vpt_idm_cfg.seed)
    extra_kwargs = dict(
        observation_shape=envs.observation_space.shape,
        action_dim=action_dim,
        input_action_dim=input_action_dim,
        continuous_actions=continuous_actions,
    )

    # lam = LatentActionModel(
    #     lam_cfg.model,
    #     key=next(rng_seq),
    #     load_from_ckpt=True,
    #     # ckpt_file=Path(config.lam_ckpt) / "model_ckpts" / "ckpt_0100.pkl",
    #     ckpt_dir=Path(config.model.lam_ckpt),
    #     **extra_kwargs,
    # )

    vpt = VPT(
        vpt_idm_cfg.model,
        key=next(rng_seq),
        load_from_ckpt=True,
        # ckpt_file=Path(config.lam_ckpt) / "model_ckpts" / "ckpt_0100.pkl",
        ckpt_dir=Path(config.model.vpt_idm_ckpt),
        **extra_kwargs,
    )

    logging.info("done loading LAM")

    # load original dataset

    logging.info("loading original dataset")

    training_env_id = config.env.env_id

    data_dir = Path(config.root_dir) / config.data.data_dir

    dataset_name = f"{config.data.dataset_name}_eid-{training_env_id}_n-{config.num_rollouts_collect}"
    file = "traj_dataset.pkl"

    data_file = (
        data_dir
        / dataset_name
        / file
    )

    jit_apply = jax.jit(vpt.model.apply, static_argnames=("self", "is_training"))

    data = np.load(data_file, allow_pickle=True)

    # replace actions with idm actions
    observations = data["observations"]

    # make it into LAPO style data, since we're loading in trajectories
    idm_actions = []
    window_size = 2 + vpt_idm_cfg.model.context_len

    for j in tqdm.tqdm(range(observations.shape[1] - window_size + 1)):
        window = observations[:, j : j + window_size]
        idm_output, _ = jit_apply(
            vpt._ts.params,
            vpt._ts.state,
            next(rng_seq),
            vpt,
            states=window.astype(jnp.float32),
            is_training=False,
        )
        idm_action = idm_output.action
        idm_actions.append(idm_action)

    idm_actions = np.stack(idm_actions, axis=1)
    print(f"idm_actions shape: {idm_actions.shape}")
    b, _ = idm_actions.shape

    # add zeros at the end
    zeros = np.zeros((b, 1))
    pre = np.zeros((b, vpt_idm_cfg.model.context_len))
    idm_actions = np.concatenate([pre, idm_actions, zeros], axis=1)

    new_data = dict(data)
    new_data["actions"] = idm_actions

    # save new data in same directory as old data
    relabel_file = data_file.parent / f"traj_dataset_vpt_idm.pkl"
    with open(relabel_file, "wb") as f:
        pickle.dump(new_data, f)


if __name__ == "__main__":
    app.run(main)
