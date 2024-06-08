import matplotlib.pyplot as plt
import jax
import tensorflow as tf
import rlds
import tensorflow_datasets as tfds
import importlib
from pathlib import Path
from absl import logging
from functools import partial
from ml_collections.config_dict import ConfigDict


def episode_to_step_procgen(episode, size):
    episode = tf.data.Dataset.from_tensor_slices(episode)
    return rlds.transformations.batch(episode, size=size, shift=1, drop_remainder=True)


def step_to_transition(step):
    transition = {
        "observations": step["observation"],
        "actions": step["action"],
        "rewards": step["reward"],
        "discount": step["discount"],
        "is_first": step["is_first"],
        "is_last": step["is_last"],
        "is_terminal": step["is_terminal"],
    }
    return transition


def episode_to_step(episode, size):
    batched_steps = rlds.transformations.batch(
        episode["steps"], size=size, shift=1, drop_remainder=True
    )
    return batched_steps.map(step_to_transition)


def convert_to_episodes(episode):
    # hacky solution to get the full episode and all the steps together
    episode = episode["steps"].batch(10000)
    episode = next(iter(episode))
    # also change the naming scheme
    episode = {
        "observations": episode["observation"],
        "actions": episode["action"],
        "rewards": episode["reward"],
        "discount": episode["discount"],
        "is_first": episode["is_first"],
        "is_last": episode["is_last"],
        "is_terminal": episode["is_terminal"],
    }
    return episode


def load_data(config: ConfigDict, rng: jax.random.PRNGKey):
    datasets = {}

    # module = importlib.import_module(config.data.dataset_name)
    data_dir = Path(config.root_dir) / "tensorflow_datasets"
    if config.env.env_name == "atari":
        ds_builder = tfds.builder(
            config.data.dataset_name + f"/{config.env.env_id}_run_1"
        )
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
        ds = tfds.load(
            config.data.dataset_name + f"/{config.env.env_id}_run_1",
            split="+".join(data_splits),
            read_config=read_config,
            data_dir=str(data_dir),
        )
        logging.info(f"dataset size: {len(ds)}")

        ds = ds.shuffle(10000, reshuffle_each_iteration=False)
        ds = ds.take(config.data.num_trajs)

        if config.data.data_type == "trajectories":
            ds = ds.map(convert_to_episodes)
            # take batches of full episodes
            # ds = ds.padded_batch(config.data.batch_size)
            ds = ds.bucket_by_sequence_length(
                lambda x: tf.shape(x["observation"])[0],
                bucket_boundaries=[500, 1000],
                bucket_batch_sizes=[16, 8, 4],
            )
        else:
            if config.data.data_type == "lapo":
                ds = ds.flat_map(
                    partial(episode_to_step, size=config.data.context_len + 2)
                )
            elif config.data.data_type == "transitions":
                ds = ds.flat_map(
                    partial(episode_to_step, size=config.data.num_frame_stack)
                )

            ds = ds.batch(config.data.batch_size, drop_remainder=True)

        num_batches = ds.reduce(0, lambda x, _: x + 1).numpy()

        ds = ds.prefetch(2)
        # ds = ds.as_numpy_iterator()

        logging.info(f"Number of batches: {num_batches}")
        datasets["train"] = ds
        datasets["val"] = {config.env.env_id: ds}

    elif config.env.env_name == "procgen":
        for split in ["train", "val"]:
            ds = tfds.load(
                config.data.dataset_name + f"/{config.env.env_id}",
                split=split,
                data_dir=str(data_dir),
            )

            # import ipdb

            # ipdb.set_trace()
            # first shuffle the episodes
            ds = ds.shuffle(10000, reshuffle_each_iteration=False)

            # and limit the number of trajectories that we use
            ds = ds.take(config.data.num_trajs)

            logging.info(f"number of trajectories in dataset: {len(ds)}")

            if config.data.data_type == "trajectories":
                ds = ds.padded_batch(config.data.batch_size)
            else:
                if config.data.data_type == "lapo":
                    ds = ds.flat_map(
                        partial(
                            episode_to_step_procgen, size=config.data.context_len + 2
                        )
                    )
                elif config.data.data_type == "transitions":
                    ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)

                ds = ds.batch(config.data.batch_size, drop_remainder=True)

            # num_batches = ds.reduce(0, lambda x, _: x + 1).numpy()
            ds = ds.prefetch(2)
            # ds = ds.as_numpy_iterator()

            # logging.info(f"Number of batches: {num_batches} for {split}")
            datasets[split] = ds
        datasets["val"] = {config.env.env_id: datasets["val"]}

    return datasets["train"], datasets["val"], datasets["val"]


if __name__ == "__main__":
    config = ConfigDict(
        {
            "root_dir": "/scr/aliang80/varibad_jax/varibad_jax",
            "env": {"env_name": "atari", "env_id": "Pong"},
            "data": {
                "num_trajs": 100,
                "batch_size": 32,
                "context_len": 2,
                "num_frame_stack": 4,
                "data_type": "transitions",
                "dataset_name": "rlu_atari_checkpoints_ordered",
            },
        }
    )
    rng = jax.random.PRNGKey(0)
    tf.random.set_seed(0)

    # logging.info("Loading transitions dataset:")
    # train_ds, val_ds, test_ds = load_data(config, rng)
    # for batch in train_ds:
    #     for k, v in batch.items():
    #         print(k, v.shape, v.min(), v.max())

    #     observations = batch["observations"]

    #     # make a grid of observations and show with plt
    #     plt.figure(figsize=(10, 10))
    #     for i in range(4):
    #         for j in range(4):
    #             plt.subplot(4, 4, i * 4 + j + 1)
    #             plt.imshow(observations[i, j], cmap="gray")
    #             plt.title(f"Frame {j}")
    #             plt.axis("off")

    #     # tight layout
    #     plt.tight_layout()
    #     plt.savefig("atari_observations.png")
    #     plt.close()
    #     break

    # logging.info("Loading LAPO dataset:")
    # config.data.data_type = "lapo"
    # train_ds, val_ds, test_ds = load_data(config, rng)
    # for batch in train_ds:
    #     for k, v in batch.items():
    #         print(k, v.shape, v.min(), v.max())
    #     break

    # logging.info("Loading trajectory dataset")
    # config.data.data_type = "trajectories"
    # train_ds, val_ds, test_ds = load_data(config, rng)
    # for batch in train_ds:
    #     for k, v in batch.items():
    #         print(k, v.shape)
    #     break

    logging.info("Loading procgen dataset")
    config.env.env_name = "procgen"
    config.env.env_id = "bigfish"
    config.data.dataset_name = "procgen_dataset"
    config.data.data_type = "lapo"

    train_ds, val_ds, test_ds = load_data(config, rng)
    for batch in train_ds:
        for k, v in batch.items():
            print(k, v.shape)

        # observations = batch["observations"]

        # # make a grid of observations and show with plt
        # plt.figure(figsize=(10, 10))
        # for i in range(16):
        #     plt.subplot(4, 4, i + 1)
        #     plt.imshow(observations[i])
        #     plt.axis("off")

        # # tight layout
        # plt.tight_layout()
        # plt.savefig("procgen_observations.png")
        # plt.close()
        break
