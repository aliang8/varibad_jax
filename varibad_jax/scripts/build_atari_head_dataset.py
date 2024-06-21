import tensorflow_datasets as tfds
import os
import atari_head_dataset

os.environ["TFDS_DATA_DIR"] = (
    "/scr/aliang80/varibad_jax/varibad_jax/tensorflow_datasets"
)

if __name__ == "__main__":
    builder = tfds.builder("atari_head_image_dataset")
    data_dir = "/scr/aliang80/varibad_jax/atari_head/"
    builder.download_and_prepare(
        download_dir=data_dir,
        download_config=tfds.download.DownloadConfig(manual_dir=data_dir),
    )

    ds = builder.as_dataset(split="train")
    ds = ds.take(1).as_numpy_iterator()

    for example in ds:
        observation, action, label = example
        print(observation.shape, observation.min(), observation.max(), label, action)
        break
