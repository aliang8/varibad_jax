"""
python3 scripts/visualize_latent_space.py \
    --config=configs/procgen/action_decoder.py:lam_agent-procgen-64x64 \
    --config.env.env_id=bigfish \
    --config.env.eval_env_ids="('bigfish',)" 

script for visualizing the latent space from offline dataset
"""

from absl import app, logging
import einops
import re
import os

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import umap
from sklearn.datasets import load_digits

from varibad_jax.utils.tfds_data_utils import load_data

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.config.experimental.set_visible_devices([], "GPU")

_CONFIG = config_flags.DEFINE_config_file("config")


def visualize_latent_space(rng_seq, data, split="train", vis_type="umap"):
    all_latent_actions = []
    all_gt_actions = []

    for batch in tqdm.tqdm(data.as_numpy_iterator()):
        all_latent_actions.append(batch["latent_actions"])
        all_gt_actions.append(batch["actions"])

    all_latent_actions = np.concatenate(all_latent_actions, axis=0)
    all_gt_actions = np.concatenate(all_gt_actions, axis=0)

    # assert they are the same size
    assert all_latent_actions.shape[0] == all_gt_actions.shape[0]

    # sample 10k points
    idx = np.random.choice(all_latent_actions.shape[0], 10_000, replace=False)
    all_latent_actions = all_latent_actions[idx]
    all_gt_actions = all_gt_actions[idx]

    # visualize latent space with t-SNE
    if vis_type == "tsne":
        tsne = TSNE(n_components=2, random_state=0)
        embedding = tsne.fit_transform(all_latent_actions)
    elif vis_type == "umap":
        embedding = umap.UMAP(n_neighbors=10, min_dist=0.1).fit_transform(
            all_latent_actions
        )
        print(embedding.shape)
    else:
        raise ValueError(f"Unknown visualization type: {vis_type}")

    # use gt actions to color code the latent space
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=all_gt_actions, cmap="viridis"
    )
    legend1 = ax.legend(*scatter.legend_elements(), title="Actions")
    ax.add_artist(legend1)

    method = vis_type.capitalize()
    plt.title(f"{method} Latent space visualization ({split})")

    # save figure
    plt.savefig(f"{method}_latent_space_{split}.png")
    plt.close()


def main(_):

    tf.random.set_seed(0)
    config = _CONFIG.value

    # load original dataset
    logging.info("loading original observation-only dataset")

    rng_seq = hk.PRNGSequence(config.seed)

    config.data.data_type = "transitions"
    config.data.num_trajs = 100_000
    config.data.batch_size = 5_000  # load faster
    config.data.load_latent_actions = True
    train_data, eval_data, _ = load_data(
        config, next(rng_seq), shuffle=False, drop_remainder=False
    )

    if config.env.env_name == "procgen":
        for split in tqdm.tqdm(["train", "val"], desc="split"):
            if split == "train":
                data = train_data
            else:
                data = eval_data["bigfish"]
            visualize_latent_space(rng_seq, data, split=split)
    else:
        visualize_latent_space(rng_seq, train_data, split="")


if __name__ == "__main__":
    app.run(main)
