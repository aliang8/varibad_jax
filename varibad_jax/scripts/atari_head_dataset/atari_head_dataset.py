import tensorflow_datasets as tfds
import tensorflow as tf
import os
import re
import math
from pathlib import Path
from .atari_head_processor import AtariHeadProcessor

os.environ["TFDS_DATA_DIR"] = (
    "/scr/aliang80/varibad_jax/varibad_jax/tensorflow_datasets"
)


def sort_numerically(file_list):
    def numerical_key(filename):
        # Extract the numerical part at the end of the filename before the file extension
        match = re.search(r"_(\d+)\.", str(filename))
        return int(match.group(1)) if match else filename

    return sorted(file_list, key=numerical_key)


class AtariHeadImageDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for atari_head dataset."""

    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="AtariHead Image dataset",
            disable_shuffling=True,
            features=tfds.features.FeaturesDict(
                {
                    "observations": tfds.features.Image(),
                    "actions": tfds.features.Tensor(shape=(), dtype=tf.int32),
                    "is_terminal": tfds.features.Tensor(shape=(), dtype=tf.int32),
                    "timestep": tfds.features.Tensor(shape=(), dtype=tf.int32),
                    # "id": tfds.features.Text(),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        # data_dir = dl_manager.manual_dir
        data_dir = Path("/scr/aliang80/varibad_jax/atari_head/")

        self.ah = AtariHeadProcessor(data_dir)

        return {
            "train": self._generate_examples(data_dir),
        }

    def _generate_examples(self, path):
        """Yields examples."""

        game = "Asterix"
        count = 0
        for trial_number in self.ah.game_trials(game):
            data = self.ah.read_gazes(trial_number)

            actions = data["action"].to_list()
            actions = [0 if math.isnan(x) else x for x in actions]
            frame_ids = data["frame_id"].to_list()

            run_names = data["run_name"].unique().tolist()

            assert len(run_names) == 1

            run_name = run_names[0]

            for class_dir in path.iterdir():
                if class_dir.is_dir() and run_name in class_dir.name:
                    class_name = class_dir.name

                    files = sort_numerically(list(class_dir.glob("*/*")))

                    print("num files", len(files))

                    for indx, image_path in enumerate(files):
                        # check frame id is in image path
                        assert frame_ids[indx] in image_path.stem

                        count += 1
                        is_terminal = 1 if indx == len(files) - 1 else 0
                        if image_path.is_file():
                            print(f"Processing {image_path}")
                            yield f"{count:06}" + "_" + image_path.stem, {
                                "observations": image_path,
                                "actions": int(actions[indx]),
                                "is_terminal": is_terminal,
                                "timestep": count - 1,
                                # "id": frame_ids[indx],
                            }
