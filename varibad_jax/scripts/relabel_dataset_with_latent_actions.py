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
import os

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path
import pickle
import einops
import re
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
from varibad_jax.models.lam.lam import LatentActionModel, LatentActionDecoder
from varibad_jax.models.vpt.vpt import VPT
from varibad_jax.utils.tfds_data_utils import load_data

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.config.experimental.set_visible_devices([], "GPU")

_CONFIG = config_flags.DEFINE_config_file("config")


def relabel_data(config, lam, rng_seq, data, split="train"):
    jit_apply = jax.jit(lam.model.apply, static_argnames=("self", "is_training"))

    all_la_quantized = []
    all_la_prequantized = []
    dones = []  # need the dones to figure out where to split

    params = jax.tree_util.tree_map(lambda x: x[0], lam._ts.params)
    state = jax.tree_util.tree_map(lambda x: x[0], lam._ts.state)

    for batch in tqdm.tqdm(data.as_numpy_iterator()):
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

        if "lam" in config.model.name:
            quantized_la = lam_output.quantize
            la = lam_output.latent_actions
        elif "vpt" in config.model.name:
            quantized_la = lam_output.action
            la = lam_output.action

            # add extra dimension
            quantized_la = einops.repeat(quantized_la, "b -> b 1")
            la = einops.repeat(la, "b -> b 1")

        all_la_quantized.append(quantized_la)
        all_la_prequantized.append(la)

    all_la_quantized = np.concatenate(all_la_quantized, axis=0)
    all_la_prequantized = np.concatenate(all_la_prequantized, axis=0)
    dones = np.array(dones)

    # split latent actions based on dones using np.split
    quantized_la = np.split(all_la_quantized, np.where(dones)[0] + 1)[:-1]
    prequantized_la = np.split(all_la_prequantized, np.where(dones)[0] + 1)[:-1]

    for i, la in enumerate(quantized_la):
        b, d = la.shape

        # add zeros at the end
        zeros = np.zeros((1, d))
        pre = np.zeros((config.model.context_len, d))
        quantized_la[i] = np.concatenate([pre, la, zeros], axis=0)
        quantized_la[i] = tf.convert_to_tensor(quantized_la[i])

        prequantized_la[i] = np.concatenate([pre, prequantized_la[i], zeros], axis=0)

    def generator():
        for quantize, la in zip(quantized_la, prequantized_la):
            yield {
                "quantize": quantize,
                "latent_action": la,
            }

    action_dim = config.model.idm.code_dim if "lam" in config.model.name else 1

    if "atari_head" in config.data.dataset_name:
        # join them together
        latent_action_trajs = tf.concat(latent_action_trajs, axis=0)
        tfds_la = tf.data.Dataset.from_generator(
            generator,
            output_signature={
                "quantize": tf.TensorSpec(shape=(action_dim,), dtype=tf.float32),
                "latent_action": tf.TensorSpec(shape=(action_dim,), dtype=tf.float32),
            },
        )
    else:
        tfds_la = tf.data.Dataset.from_generator(
            generator,
            output_signature={
                "quantize": tf.TensorSpec(shape=(None, action_dim), dtype=tf.float32),
                "latent_action": tf.TensorSpec(
                    shape=(None, action_dim), dtype=tf.float32
                ),
            },
        )

    # concatenate and save as tfds dataset
    # latent_actions = np.concatenate(latent_action_trajs, axis=0)
    # tfds_la = tf.data.Dataset.from_tensors(latent_actions)

    # save tfds somewhere

    if "atari_head" in config.data.dataset_name:
        base = ""
    elif config.env.env_name == "atari":
        base = f"{config.env.env_id}_run_1"
    elif config.env.env_name == "procgen":
        base = f"{config.env.env_id}"

    save_dir = (
        Path(config.root_dir) / "tensorflow_datasets" / config.data.dataset_name / base
    )

    logging.info(f"save_dir: {save_dir}")
    save_file = (
        save_dir / f"la-{split}_m-{config.model.name}_nt-{config.data.num_trajs}"
    )

    tf.data.experimental.save(tfds_la, str(save_file))


def main(_):

    tf.random.set_seed(0)
    config = _CONFIG.value

    if "lam" in config.model.name or "latent" in config.model.name:
        cfg_path = Path(config.model.lam_ckpt) / "config.json"
    elif "vpt" in config.model.name:
        cfg_path = Path(config.model.vpt_idm_ckpt) / "config.json"
    else:
        raise ValueError(f"model name {config.model.name} not recognized")

    # load pretrained IDM/FDM for predicting the latent actions from observations
    logging.info("loading IDM")
    with open(cfg_path, "r") as f:
        model_cfg = ConfigDict(json.load(f))

    model_cfg = ConfigDict(model_cfg)
    if model_cfg.env.env_name == "atari":
        envs = make_atari_envs(num_envs=1, env_id=model_cfg.env.env_id)
    elif model_cfg.env.env_name == "procgen":
        envs = make_procgen_envs(num_envs=1, env_id=model_cfg.env.env_id, gamma=1)

    logging.info(f"env id: {model_cfg.env.env_id}, env name: {model_cfg.env.env_name}")
    # continuous_actions = not isinstance(
    #     envs.action_space, gym.spaces.Discrete
    # ) and not isinstance(envs.action_space, gymnax.environments.spaces.Discrete)
    # if continuous_actions:
    #     input_action_dim = action_dim = envs.action_space.shape[0]
    # else:
    continuous_actions = False
    action_dim = envs.action_space.n
    input_action_dim = 1

    if model_cfg.env.env_name == "atari":
        observation_space = envs.observation_space.shape
    elif model_cfg.env.env_name == "procgen":
        observation_space = envs.observation_space["rgb"].shape

    logging.info(f"observation shape: {observation_space}")

    rng_seq = hk.PRNGSequence(model_cfg.seed)
    extra_kwargs = dict(
        observation_shape=observation_space,
        action_dim=action_dim,
        input_action_dim=input_action_dim,
        continuous_actions=continuous_actions,
    )

    if "lam" in config.model.name or "latent" in config.model.name:
        idm = LatentActionModel(
            model_cfg.model,
            key=next(rng_seq),
            load_from_ckpt=True,
            # ckpt_file=Path(config.model.lam_ckpt) / "model_ckpts" / "ckpt_25000.pkl",
            ckpt_dir=Path(config.model.lam_ckpt),
            **extra_kwargs,
        )
        logging.info("done loading LAM")
    elif "vpt" in config.model.name:
        idm = VPT(
            model_cfg.model,
            key=next(rng_seq),
            load_from_ckpt=True,
            ckpt_dir=Path(config.model.vpt_idm_ckpt),
            **extra_kwargs,
        )
        logging.info("done loading VPT IDM")

    # load original dataset
    logging.info("loading original observation-only dataset")

    config.data.data_type = "lapo"
    config.data.num_trajs = 100_000
    config.data.load_latent_actions = False
    config.data.batch_size = 5_000

    train_data, eval_data, _ = load_data(
        config, next(rng_seq), shuffle=False, drop_remainder=False
    )

    if config.env.env_name == "procgen":
        for split in tqdm.tqdm(["train", "val"], desc="split"):
            if split == "train":
                data = train_data
            else:
                data = eval_data[config.env.env_id]
            relabel_data(model_cfg, idm, rng_seq, data, split=split)
    else:
        relabel_data(model_cfg, idm, rng_seq, train_data, split="")


if __name__ == "__main__":
    app.run(main)
