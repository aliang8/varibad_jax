import rlds
import argparse
import tqdm
import importlib
import os
from functools import partial

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress debug warning messages
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import wandb
import tensorflow as tf

WANDB_ENTITY = None
WANDB_PROJECT = "vis_rlds"

tf.random.set_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", help="name of the dataset to visualize")
args = parser.parse_args()

if WANDB_ENTITY is not None:
    render_wandb = True
    wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT)
else:
    render_wandb = False


# create TF dataset
dataset_name = args.dataset_name
print(f"Visualizing data from dataset: {dataset_name}")
module = importlib.import_module(dataset_name)
ds = tfds.load(
    dataset_name + "/bigfish",
    split="train",
    data_dir="/scr/aliang80/varibad_jax/varibad_jax/tensorflow_datasets",
)
ds = ds.shuffle(10000, reshuffle_each_iteration=False)
# ds = ds.take(1)
# print(ds.cardinality())
# print(len(ds))
# bs = 128


# # ds = ds.padded_batch(3)
# # ds = ds.ragged_batch(bs)
# # ds = (
# #     ds.flat_map(tf.data.Dataset.from_tensor_slices).shuffle(10000).prefetch(2).batch(bs)
# # )

# # ds = ds.as_numpy_iterator()


# def episode_to_step(episode, size):
#     episode = tf.data.Dataset.from_tensor_slices(episode)
#     return rlds.transformations.batch(episode, size=size, shift=1, drop_remainder=True)


# # import ipdb

# # ipdb.set_trace()

# ds = ds.flat_map(partial(episode_to_step, size=2)).batch(bs).shuffle(100).take(10)
# # ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)

# # ds = rlds.transformations.batch(ds, size=bs, shift=1)
# print(ds.cardinality())
count = 0
for i, batch in enumerate(ds.as_numpy_iterator()):
    # for k in batch:
    # import ipdb

    # ipdb.set_trace()
    # # for k, v in batch.items():
    # #     print(k, v.shape)
    # print(batch["action"].shape)
    count += 1

print(count)

count = 0

for i, batch in enumerate(ds.as_numpy_iterator()):
    # for k in batch:
    # import ipdb

    # ipdb.set_trace()
    # for k, v in batch.items():
    #     print(k, v.shape)
    # print(batch["action"].shape)
    count += 1

print(count)
