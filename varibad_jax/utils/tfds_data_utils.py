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
from varibad_jax.utils.randaugment import randaugment
import os

os.environ["TFDS_DATA_DIR"] = (
    "/scr/aliang80/varibad_jax/varibad_jax/tensorflow_datasets"
)


def episode_to_step_procgen(episode, size):
    episode = tf.data.Dataset.from_tensor_slices(episode)
    return rlds.transformations.batch(episode, size=size, shift=1, drop_remainder=True)


def step_to_transition(step, latent_action=None):
    transition = {
        "observations": step["observation"],
        "actions": step["action"],
        "rewards": step["reward"],
        "discount": step["discount"],
        "is_first": step["is_first"],
        "is_last": step["is_last"],
        "is_terminal": step["is_terminal"],
    }
    if latent_action is not None:
        transition["latent_actions"] = latent_action
    return transition


def episode_to_step(episode, size, element_spec, drop_remainder=True):
    # copy first step in episode["steps"]
    # zeros_dataset = rlds.transformations.zero_dataset_like(
    #     element_spec[rlds.STEPS]
    # ).repeat(
    #     3
    # )  # this is to account for the frame stacking

    episode_steps = episode["steps"]
    # first_step = episode["steps"].take(1)
    # first_step = first_step.repeat(3)
    # episode_steps = first_step.concatenate(episode_steps)
    if "latent_actions" in episode:
        latent_actions = episode["latent_actions"]
        latent_actions = tf.data.Dataset.from_tensor_slices(latent_actions)
        episode_steps = tf.data.Dataset.zip((episode_steps, latent_actions))

    batched_steps = rlds.transformations.batch(
        episode_steps, size=size, shift=1, drop_remainder=drop_remainder
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


def rand_crop_video(seeds, video, width, height, wiggle):
    """Random crop of a video. Assuming height < width."""
    x_wiggle = wiggle
    crop_width = height - wiggle
    y_wiggle = width - crop_width
    xx = tf.random.stateless_uniform(
        [], seed=seeds[0], minval=0, maxval=x_wiggle, dtype=tf.int32
    )
    yy = tf.random.stateless_uniform(
        [], seed=seeds[1], minval=0, maxval=y_wiggle, dtype=tf.int32
    )
    return video[:, xx : xx + crop_width, yy : yy + crop_width, :]


def rand_crop_img(seeds, img, width, height, wiggle):
    """Random crop of a img. Assuming height < width."""
    x_wiggle = wiggle
    crop_width = height - wiggle
    y_wiggle = width - crop_width
    xx = tf.random.stateless_uniform(
        [], seed=seeds[0], minval=0, maxval=x_wiggle, dtype=tf.int32
    )
    yy = tf.random.stateless_uniform(
        [], seed=seeds[1], minval=0, maxval=y_wiggle, dtype=tf.int32
    )
    return img[xx : xx + crop_width, yy : yy + crop_width, :]


def apply_image_augmentations(dataset, augmentations=[]):
    """Augment dataset with a list of augmentations."""

    def augment(seeds, features):
        # observations = tf.cast(features["observations"], tf.uint8)
        # for aug_fn in augmentations:
        #     observations = aug_fn(seeds, observations)
        observations = features["observations"]

        # need to convert to int32 first
        observations = (observations + 0.5) * 255.0
        observations = tf.cast(observations, tf.uint8)

        # apply randaugment to each observation
        if observations.ndim > 3:
            observations = tf.map_fn(
                partial(randaugment, num_layers=1, magnitude=10, seeds=seeds),
                observations,
            )
        else:
            observations = randaugment(
                observations, num_layers=1, magnitude=10, seeds=seeds
            )

        # convert back to float32
        observations = tf.cast(observations, tf.float32) / 255.0 - 0.5

        features["observations"] = observations
        return features

    randds = tf.data.experimental.RandomDataset(1).batch(2).batch(4)
    dataset = tf.data.Dataset.zip((randds, dataset))
    dataset = dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def load_data(
    config: ConfigDict,
    rng: jax.random.PRNGKey,
    shuffle: bool = True,
    drop_remainder: bool = True,
    channel_first: bool = False,
):
    datasets = {}
    data_dir = Path(config.root_dir) / "tensorflow_datasets"

    # add a new field for mask
    def add_mask(x):
        x["mask"] = tf.ones_like(x["actions"])
        x["timestep"] = tf.range(tf.shape(x["actions"])[0])
        return x

    if "atari_head" in config.data.dataset_name:
        ds_builder = tfds.builder(config.data.dataset_name)
        ds = ds_builder.as_dataset(split="train", shuffle_files=False)
        print("number of examples: ", len(ds))

        if config.data.load_latent_actions:
            la_file = data_dir / config.data.dataset_name / "latent_actions_"
            # this is a tf.data.Dataset
            latent_actions = tf.data.experimental.load(str(la_file))
            ds = tf.data.Dataset.zip((ds, latent_actions))
            logging.info(f"Loaded latent actions from {la_file}")

            def combine_latent_actions(x, la):
                x["latent_actions"] = la["quantize"]
                return x

            ds = ds.map(combine_latent_actions)

        if config.data.data_type == "trajectories":
            pass
        elif config.data.data_type in ["lapo", "transitions"]:
            if config.data.data_type == "lapo":
                ds = rlds.transformations.batch(
                    ds, size=config.data.context_len + 2, shift=1, drop_remainder=True
                )
            elif config.data.data_type == "transitions":
                pass

            def reshape_obs(x):
                if channel_first:
                    x["observations"] = tf.transpose(x["observations"], [2, 0, 1])

                # also resize to 64x64
                x["observations"] = tf.image.resize(x["observations"], [64, 64]) / 255.0
                return x

            # if config.data.image_augmentations:
            #     ds = apply_image_augmentations(ds)
            ds = ds.map(reshape_obs)
            ds = ds.map(add_mask)

            if shuffle:
                ds = ds.shuffle(10000, reshuffle_each_iteration=True)

            ds = ds.batch(config.data.batch_size, drop_remainder=drop_remainder)

        datasets["train"] = ds
        datasets["val"] = {config.env.env_id: ds}
    elif config.env.env_name == "atari":
        env = config.env.env_id
        env_base = env.split("-")[0]
        ds_name = config.data.dataset_name + f"/{env_base}_run_1"

        ds_builder = tfds.builder(ds_name)

        if config.data.load_latent_actions:
            la_file = data_dir / ds_name / "la"
            # this is a tf.data.Dataset
            latent_actions = tf.data.experimental.load(str(la_file))
            logging.info(f"Loaded latent actions from {la_file}")

        data_splits = []
        # data_percent = 100
        # for split, info in ds_builder.info.splits.items():
        #     # Convert `data_percent` to number of episodes to allow
        #     # for fractional percentages.
        #     num_episodes = int((data_percent / 100) * info.num_examples)
        #     if num_episodes == 0:
        #         raise ValueError(f"{data_percent}% leads to 0 episodes in {split}!")
        #     # Sample first `data_percent` episodes from each of the data split
        #     data_splits.append(f"{split}[:{num_episodes}]")
        #     # Interleave episodes across different splits/checkpoints
        #     # Set `shuffle_files=True` to shuffle episodes across files within splits

        # since we are focusing on BC, we only take episodes from the final checkpoints
        for split in ["checkpoint_48", "checkpoint_49"]:
            info = ds_builder.info.splits[split]
            num_episodes = info.num_examples
            data_splits.append(f"{split}[:{num_episodes}]")

        read_config = tfds.ReadConfig(
            interleave_cycle_length=len(data_splits),
            shuffle_reshuffle_each_iteration=True,
            enable_ordering_guard=False,
        )
        ds = tfds.load(
            ds_name,
            split="+".join(data_splits),
            read_config=read_config,
            data_dir=str(data_dir),
        )
        logging.info(f"number of trajectories in the dataset: {len(ds)}")

        # first let's combine the latent actions with the dataset
        if config.data.load_latent_actions:
            ds = tf.data.Dataset.zip((ds, latent_actions))

            def combine_latent_actions(x, la):
                x["latent_actions"] = la["quantize"]
                return x

            ds = ds.map(combine_latent_actions)

        if shuffle:
            # first shuffle the episodes
            ds = ds.shuffle(10000, reshuffle_each_iteration=False)

        ds = ds.take(config.data.num_trajs)
        ds = ds.map(add_mask)

        if config.data.data_type == "trajectories":
            ds = ds.map(convert_to_episodes)
            # take batches of full episodes
            # ds = ds.padded_batch(config.data.batch_size)
            ds = ds.bucket_by_sequence_length(
                lambda x: tf.shape(x["observations"])[0],
                bucket_boundaries=[500, 1000],
                bucket_batch_sizes=[16, 8, 4],
            )
        elif config.data.data_type in ["lapo", "transitions"]:
            if config.data.data_type == "lapo":
                ds = ds.flat_map(
                    partial(
                        episode_to_step,
                        size=config.data.context_len + 2,
                        element_spec=ds.element_spec,
                        drop_remainder=True,
                    )
                )

                def reshape_obs(x):
                    # x["observations"] = tf.squeeze(x["observations"], axis=-1)
                    x["observations"] = tf.cast(x["observations"], tf.float64) / 255.0

                    # let's resize the images to 64x64
                    x["observations"] = tf.image.resize(x["observations"], [64, 64])

                    if channel_first:
                        x["observations"] = tf.transpose(
                            x["observations"], [0, 3, 1, 2]
                        )
                    return x

                # apply augmentations
                ds = augment_dataset(ds)

                ds = ds.map(reshape_obs)

            elif config.data.data_type == "transitions":
                ds = ds.flat_map(
                    partial(
                        episode_to_step,
                        size=config.data.num_frame_stack,
                        element_spec=ds.element_spec,
                        drop_remainder=True,
                    )
                )

                def reshape_obs(x):

                    # import ipdb

                    # ipdb.set_trace()
                    # to get the framestack
                    # squeeze last dimension
                    x["observations"] = tf.squeeze(x["observations"], axis=-1)
                    # change dtype
                    x["observations"] = tf.cast(x["observations"], tf.float64)
                    x["observations"] = (
                        tf.transpose(x["observations"], [1, 2, 0]) / 255.0
                    )

                    if channel_first:
                        x["observations"] = tf.transpose(x["observations"], [2, 0, 1])

                    # but take the first for every other key
                    for k in x.keys():
                        if k != "observations":
                            x[k] = x[k][0]
                    return x

                # swapaxes
                ds = ds.map(reshape_obs)

            ds = ds.batch(config.data.batch_size, drop_remainder=drop_remainder)
        else:
            raise ValueError(f"Data type {config.data.data_type} not recognized")

        # num_batches = ds.reduce(0, lambda x, _: x + 1).numpy()

        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        # ds = ds.as_numpy_iterator()

        # logging.info(f"Number of batches: {num_batches}")
        datasets["train"] = ds
        datasets["val"] = {config.env.env_id: ds}

    elif config.env.env_name == "procgen":
        for split in ["train", "val"]:
            ds_name = f"{config.data.dataset_name}/{config.env.env_id}"

            ds = tfds.load(ds_name, split=split, data_dir=str(data_dir))

            # for i, traj in enumerate(ds):
            #     print(
            #         i, traj["is_terminal"].numpy().sum(), traj["observations"].shape[0]
            #     )
            #     if not traj["is_terminal"].numpy().sum():
            #         print(i, traj["observations"].shape[0])
            #         break
            #     # if traj["observations"].shape[0] < 2:
            #     #     import ipdb

            #     #     ipdb.set_trace()
            #     print(traj["observations"].shape[0])

            def filter_fn(traj):
                return tf.math.greater(tf.shape(traj["observations"])[0], 2)

            ds = ds.filter(filter_fn)

            if config.data.load_latent_actions:
                model = config.model.name.split("_")[0]

                if "vpt" in config.model.name:
                    la_file = (
                        data_dir
                        / ds_name
                        / f"la-{split}_m-{model}_nt-{config.model.idm_nt}"
                    )
                else:
                    la_file = data_dir / ds_name / f"la-{split}"

                # this is a tf.data.Dataset
                latent_actions = tf.data.experimental.load(str(la_file))
                logging.info(f"Loaded latent actions from {la_file}")

            if config.data.load_latent_actions:
                ds = tf.data.Dataset.zip((ds, latent_actions))

                def combine_latent_actions(x, la):
                    x["latent_actions"] = la["quantize"]
                    return x

                ds = ds.map(combine_latent_actions)

            ds = ds.cache()
            # first shuffle the episodes
            if shuffle:
                ds = ds.shuffle(100000, reshuffle_each_iteration=False)

            # and limit the number of trajectories that we use
            if split == "train":
                ds = ds.take(config.data.num_trajs)

            ds = ds.map(add_mask)

            # logging.info(f"number of trajectories in dataset: {len(ds)}")

            if config.data.data_type == "trajectories":

                # ds = ds.padded_batch(config.data.batch_size)
                ds = ds.bucket_by_sequence_length(
                    lambda x: tf.shape(x["observations"])[0],
                    bucket_boundaries=[500, 1000, 3000],
                    bucket_batch_sizes=[2, 2, 1, 1],
                    pad_to_bucket_boundary=True,
                    drop_remainder=True,
                )
            else:
                if config.data.data_type == "lapo":
                    ds = ds.flat_map(
                        partial(
                            episode_to_step_procgen, size=config.data.context_len + 2
                        )
                    )
                elif config.data.data_type == "transitions":
                    ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)

                def reshape_obs(x):
                    if channel_first:
                        x["observations"] = tf.transpose(
                            x["observations"], [0, 3, 1, 2]
                        )
                    return x

                if config.data.image_augmentations:
                    ds = apply_image_augmentations(ds)
                ds = ds.map(reshape_obs)

                # ds = ds.cache()
                ds = ds.batch(config.data.batch_size, drop_remainder=drop_remainder)

            # num_batches = ds.reduce(0, lambda x, _: x + 1).numpy()
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            # ds = ds.as_numpy_iterator()

            # logging.info(f"Number of batches: {num_batches} for {split}")
            datasets[split] = ds
        datasets["val"] = {config.env.env_id: datasets["val"]}

    return datasets["train"], datasets["val"], datasets["val"]


if __name__ == "__main__":
    # config = ConfigDict(
    #     {
    #         "root_dir": "/scr/aliang80/varibad_jax/varibad_jax",
    #         "env": {"env_name": "atari", "env_id": "Pong"},
    #         "data": {
    #             "load_latent_actions": True,
    #             "num_trajs": 5,
    #             "batch_size": 32,
    #             "context_len": 1,
    #             "num_frame_stack": 4,
    #             "data_type": "transitions",
    #             "dataset_name": "rlu_atari_checkpoints_ordered",
    #         },
    #     }
    # )
    # config = ConfigDict(
    #     {
    #         "root_dir": "/scr/aliang80/varibad_jax/varibad_jax",
    #         "env": {"env_name": "procgen", "env_id": "bigfish"},
    #         "data": {
    #             "load_latent_actions": True,
    #             "num_trajs": 5,
    #             "batch_size": 32,
    #             "context_len": 1,
    #             "num_frame_stack": 4,
    #             "data_type": "transitions",
    #             "dataset_name": "procgen_dataset",
    #         },
    #     }
    # )
    config = ConfigDict(
        {
            "root_dir": "/scr/aliang80/varibad_jax/varibad_jax",
            "env": {"env_name": "atari_head", "env_id": "Asterix"},
            "data": {
                "load_latent_actions": True,
                "num_trajs": 5,
                "batch_size": 32,
                "context_len": 1,
                "num_frame_stack": 4,
                "data_type": "transitions",
                "dataset_name": "atari_head_image_dataset",
            },
        }
    )
    rng = jax.random.PRNGKey(0)
    tf.random.set_seed(0)

    logging.info("Loading transitions dataset:")
    train_ds, val_ds, test_ds = load_data(config, rng)
    for batch in train_ds.as_numpy_iterator():
        for k, v in batch.items():
            print(k, v.shape)

        observations = batch["observations"]

        # make a grid of observations and show with plt
        plt.figure(figsize=(10, 10))
        for i in range(16):
            # for j in range(4):
            # plt.subplot(4, 4, i * 4 + j + 1)
            # plt.imshow(observations[i, ..., j])
            # plt.title(f"Frame {j}")
            # plt.axis("off")
            plt.subplot(4, 4, i + 1)
            plt.imshow(observations[i])
            plt.axis("off")

        # tight layout
        plt.tight_layout()
        plt.savefig(f"{config.env.env_id}_observations.png")
        plt.close()
        break

    # logging.info("Loading LAPO dataset:")
    # config.data.data_type = "lapo"
    # train_ds, val_ds, test_ds = load_data(config, rng)
    # for batch in train_ds:
    #     # print(batch)
    #     for k, v in batch.items():
    #         print(k, v.shape)

    #     observations = batch["observations"][0]
    #     print(observations.shape)
    #     plt.figure(figsize=(10, 10))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(observations[0])
    #     plt.axis("off")
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(observations[1])
    #     plt.axis("off")
    #     plt.tight_layout()
    #     plt.savefig(f"{config.env.env_id}_observations_lapo.png")
    #     plt.close()
    #     break

    # logging.info("Loading trajectory dataset")
    # config.data.data_type = "trajectories"
    # train_ds, val_ds, test_ds = load_data(config, rng)
    # for batch in train_ds:
    #     for k, v in batch.items():
    #         print(k, v.shape)
    #     import ipdb

    #     ipdb.set_trace()
    #     break

    # logging.info("Loading procgen dataset")
    # config.env.env_name = "procgen"
    # config.env.env_id = "bigfish"
    # config.data.dataset_name = "procgen_dataset"
    # config.data.data_type = "lapo"

    # train_ds, val_ds, test_ds = load_data(config, rng)
    # for batch in train_ds:
    #     for k, v in batch.items():
    #         print(k, v.shape)

    #     # observations = batch["observations"]

    #     # # make a grid of observations and show with plt
    #     # plt.figure(figsize=(10, 10))
    #     # for i in range(16):
    #     #     plt.subplot(4, 4, i + 1)
    #     #     plt.imshow(observations[i])
    #     #     plt.axis("off")

    #     # # tight layout
    #     # plt.tight_layout()
    #     # plt.savefig("procgen_observations.png")
    #     # plt.close()
    #     break
