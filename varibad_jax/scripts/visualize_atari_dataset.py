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
# module = importlib.import_module(dataset_name)


ds_builder = tfds.builder(dataset_name)
data_splits = []
data_percent = 1.0
for split, info in ds_builder.info.splits.items():
    # Convert `data_percent` to number of episodes to allow
    # for fractional percentages.
    num_episodes = int((data_percent / 100) * info.num_examples)
    if num_episodes == 0:
        raise ValueError(f"{data_percent}% leads to 0 episodes in {split}!")
    # Sample first `data_percent` episodes from each of the data split
    data_splits.append(f"{split}[:{num_episodes}]")
    # Interleave episodes across different splits/checkpoints
    # Set `shuffle_files=True` to shuffle episodes across files within splits
read_config = tfds.ReadConfig(
    interleave_cycle_length=len(data_splits),
    shuffle_reshuffle_each_iteration=True,
    enable_ordering_guard=False,
)
# print(data_splits)
ds = tfds.load(
    dataset_name,
    split="+".join(data_splits),
    read_config=read_config,
    # split="checkpoint_40",
    data_dir="/scr/aliang80/varibad_jax/varibad_jax/tensorflow_datasets",
)
# ds = tf.data.Dataset.concatenate(ds["checkpoint_40"], ds["checkpoint_41"])
# ds = tf.data.Dataset.from_tensor_slices(ds.values())
# concat_ds = ds.interleave(
#     lambda x: x,
#     cycle_length=1,
#     num_parallel_calls=tf.data.AUTOTUNE,
# )

# ds = ds.take(1)
# print(ds.cardinality())
# print(len(ds))
# bs = 128


def step_to_transition(step):
    transition = {
        "observation": step["observation"],
        "action": step["action"],
        "reward": step["reward"],
        "discount": step["discount"],
        "is_first": step["is_first"],
        "is_last": step["is_last"],
        "is_terminal": step["is_terminal"],
    }
    return transition


def episode_to_step(episode, size=1):
    # episode["steps"] = tf.data.Dataset.from_tensor_slices(episode["steps"])
    batched_steps = rlds.transformations.batch(
        episode["steps"], size=size, shift=1, drop_remainder=True
    )
    ds = batched_steps.map(step_to_transition)
    return ds


ds = ds.flat_map(episode_to_step).batch(2)
# ds = ds.batch(256)

# ds = ds.padded_batch(3)
# ds = ds.ragged_batch(bs)
# ds = (
#     ds.flat_map(tf.data.Dataset.from_tensor_slices).shuffle(10000).prefetch(2).batch(bs)
# )

# ds = ds.as_numpy_iterator()


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
for i, batch in enumerate(ds):
    import ipdb

    ipdb.set_trace()
    # for k in batch:
    # for k, v in batch.items():
    #     print(k, v.shape)
    # print(batch["action"].shape)
    count += 1

print(count)

# for i, batch in enumerate(ds):
#     # for k in batch:
#     # import ipdb

#     # ipdb.set_trace()
#     # for k, v in batch.items():
#     #     print(k, v.shape)
#     print(batch["action"].shape)
