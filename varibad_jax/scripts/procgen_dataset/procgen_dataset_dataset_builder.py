"""
export CUDA_VISIBLE_DEVICES=
tfds build --data_dir=/scr/aliang80/varibad_jax/varibad_jax/tensorflow_datasets --overwrite --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=10"



tfds build --data_dir=/scr/aliang80/varibad_jax/varibad_jax/tensorflow_datasets/procgen2_dataset --overwrite --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=10"
"""

import os

os.environ["TFDS_DATA_DIR"] = (
    "/scr/aliang80/varibad_jax/varibad_jax/tensorflow_datasets"
)
from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


class Bossfight(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for procgen dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "observations": tfds.features.Tensor(
                        shape=(None, 64, 64, 3),
                        dtype=np.float32,
                        doc="RGB image of the environment normalized.",
                    ),
                    "actions": tfds.features.Tensor(
                        shape=(None,),
                        dtype=np.int32,
                        doc="Discrete actions.",
                    ),
                    "discount": tfds.features.Tensor(
                        shape=(None,),
                        dtype=np.float32,
                        doc="Discount if provided, default to 1.",
                    ),
                    "rewards": tfds.features.Tensor(
                        shape=(None,),
                        dtype=np.float32,
                        doc="Reward if provided, 1 on final step for demos.",
                    ),
                    "is_first": tfds.features.Tensor(
                        shape=(None,),
                        dtype=np.bool_,
                        doc="True on first step of the episode.",
                    ),
                    "is_last": tfds.features.Tensor(
                        shape=(None,),
                        dtype=np.bool_,
                        doc="True on last step of the episode.",
                    ),
                    "is_terminal": tfds.features.Tensor(
                        shape=(None,),
                        dtype=np.bool_,
                        doc="True on last step of the episode if it is a terminal step, True for demos.",
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            "train": self._generate_examples(
                path="/scr/aliang80/varibad_jax/varibad_jax/datasets/procgen/expert_data/bossfight/train/trajs/*.npz"
            ),
            "val": self._generate_examples(
                path="/scr/aliang80/varibad_jax/varibad_jax/datasets/procgen/expert_data/bossfight/test/trajs/*.npz"
            ),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(
                episode_path, allow_pickle=True
            )  # this is a list of dicts in our case

            num_steps = len(data["observations"])

            observations = data["observations"]
            actions = data["actions"]
            rewards = data["rewards"]

            sample = {
                "observations": [],
                "actions": [],
                "discount": [],
                "rewards": [],
                "is_first": [],
                "is_last": [],
                "is_terminal": [],
            }

            for step in range(num_steps):
                sample["observations"].append(observations[step])
                sample["actions"].append(actions[step])
                sample["rewards"].append(rewards[step])
                sample["discount"].append(1.0)
                sample["is_first"].append(step == 0)
                sample["is_last"].append(step == num_steps - 1)
                sample["is_terminal"].append(step == num_steps - 1)

            # create output data sample
            # sample = {"steps": episode, "episode_metadata": {"file_path": episode_path}}

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)[:5000]
        print(f"len episode_paths: {len(episode_paths)}")

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return beam.Create(episode_paths) | beam.Map(_parse_example)
