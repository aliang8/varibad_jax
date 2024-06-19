"""
python3 scripts/relabel_dataset_with_latent_actions_atari.py \
    --config=configs/atari/action_decoder.py:lam_agent-atari-64x64 \
    --config.env.env_id=Pong \
    --config.env.eval_env_ids="('Pong',)" \
    --config.model.lam_ckpt=/scr/aliang80/varibad_jax/varibad_jax/results/lam/al-False/nt-100000/eid-Pong/en-atari/b-0.05/code_d-128/n_codes-64/usd-False 

script for relabelling atari dataset with latent actions 
important that we use the same random seed here and in the data loading 

should take around 10 minutes to run for the atari trajectories 
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
import tensorflow_datasets as tfds
import tensorflow as tf
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags

from varibad_jax.envs.utils import make_envs, make_procgen_envs, make_atari_envs
from varibad_jax.utils.rollout import run_rollouts
from varibad_jax.models.varibad.varibad import VariBADModel
from varibad_jax.agents.ppo.ppo import PPOAgent
from varibad_jax.models.base import RandomActionAgent
from varibad_jax.models.lam.lam import LatentActionModel, LatentActionDecoder
from varibad_jax.utils.tfds_data_utils import load_data

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.config.experimental.set_visible_devices([], "GPU")

_CONFIG = config_flags.DEFINE_config_file("config")


def main(_):

    tf.random.set_seed(0)
    config = _CONFIG.value

    # save_dir = (
    #     Path(config.root_dir)
    #     / "tensorflow_datasets"
    #     / config.data.dataset_name
    #     / f"{config.env.env_id}_run_1"
    # )
    # save_file = save_dir / "latent_actions"

    # if save_file.exists():
    #     tfds_la = tf.data.experimental.load(str(save_file))
    #     for la in tfds_la:
    #         print(la.shape)

    #     import ipdb

    #     ipdb.set_trace()

    # load pretrained IDM/FDM for predicting the latent actions from observations
    logging.info("loading LAM")
    lam_cfg_path = Path(config.model.lam_ckpt) / "config.json"
    with open(lam_cfg_path, "r") as f:
        lam_cfg = ConfigDict(json.load(f))

    lam_cfg = ConfigDict(lam_cfg)
    if lam_cfg.env.env_name == "atari":
        envs = make_atari_envs(num_envs=1, env_id=lam_cfg.env.env_id)
    elif lam_cfg.env.env_name == "procgen":
        envs = make_procgen_envs(num_envs=1, env_id=lam_cfg.env.env_id, gamma=1)

    logging.info(f"env id: {lam_cfg.env.env_id}, env name: {lam_cfg.env.env_name}")
    # continuous_actions = not isinstance(
    #     envs.action_space, gym.spaces.Discrete
    # ) and not isinstance(envs.action_space, gymnax.environments.spaces.Discrete)
    # if continuous_actions:
    #     input_action_dim = action_dim = envs.action_space.shape[0]
    # else:
    continuous_actions = False
    action_dim = envs.action_space.n
    input_action_dim = 1

    if lam_cfg.env.env_name == "atari":
        observation_space = envs.observation_space.shape
    elif lam_cfg.env.env_name == "procgen":
        observation_space = envs.observation_space["rgb"].shape

    logging.info(f"observation shape: {observation_space}")

    rng_seq = hk.PRNGSequence(lam_cfg.seed)
    extra_kwargs = dict(
        observation_shape=observation_space,
        action_dim=action_dim,
        input_action_dim=input_action_dim,
        continuous_actions=continuous_actions,
    )

    lam = LatentActionModel(
        lam_cfg.model,
        key=next(rng_seq),
        load_from_ckpt=True,
        # ckpt_file=Path(config.model.lam_ckpt) / "model_ckpts" / "ckpt_0008.pkl",
        ckpt_dir=Path(config.model.lam_ckpt),
        **extra_kwargs,
    )
    logging.info("done loading LAM")

    jit_apply = jax.jit(lam.model.apply, static_argnames=("self", "is_training"))

    # load original dataset
    logging.info("loading original observation-only dataset")

    all_latent_actions = []
    dones = []  # need the dones to figure out where to split

    config.data.data_type = "lapo"
    config.data.num_trajs = 100_000
    config.data.load_latent_actions = False
    train_data, eval_data, _ = load_data(
        config, next(rng_seq), shuffle=False, drop_remainder=False
    )

    params = jax.tree_util.tree_map(lambda x: x[0], lam._ts.params)
    state = jax.tree_util.tree_map(lambda x: x[0], lam._ts.state)

    for batch in tqdm.tqdm(train_data.as_numpy_iterator()):
        observations = batch["observations"]
        done = batch["is_terminal"]
        dones.extend(done[:, -1])

        lam_output, _ = jit_apply(
            params,
            state,
            next(rng_seq),
            lam,
            states=observations.astype(jnp.float32),
            is_training=False,
        )
        latent_action = lam_output.quantize
        all_latent_actions.append(latent_action)

    all_latent_actions = np.concatenate(all_latent_actions, axis=0)
    dones = np.array(dones)

    # split latent actions based on dones using np.split
    latent_action_trajs = np.split(all_latent_actions, np.where(dones)[0] + 1)[:-1]

    for i, la in enumerate(latent_action_trajs):
        b, d = la.shape

        # add zeros at the end
        zeros = np.zeros((1, d))
        pre = np.zeros((lam_cfg.model.context_len, d))
        latent_action_trajs[i] = np.concatenate([pre, la, zeros], axis=0)
        latent_action_trajs[i] = tf.convert_to_tensor(latent_action_trajs[i])

    def generator():
        for la in latent_action_trajs:
            yield la

    tfds_la = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(
            shape=(None, lam_cfg.model.idm.code_dim), dtype=tf.float32
        ),
    )

    # concatenate and save as tfds dataset
    # latent_actions = np.concatenate(latent_action_trajs, axis=0)
    # tfds_la = tf.data.Dataset.from_tensors(latent_actions)

    # save tfds somewhere

    if lam_cfg.env.env_name == "atari":
        base = f"{config.env.env_id}_run_1"
    elif lam_cfg.env.env_name == "procgen":
        base = f"{config.env.env_id}"

    save_dir = (
        Path(config.root_dir) / "tensorflow_datasets" / config.data.dataset_name / base
    )

    logging.info(f"save_dir: {save_dir}")
    save_file = save_dir / "latent_actions_train"

    tf.data.experimental.save(tfds_la, str(save_file))


if __name__ == "__main__":
    app.run(main)
