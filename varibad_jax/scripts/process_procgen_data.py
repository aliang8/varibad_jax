import time
import pickle
import json
import numpy as np
from pathlib import Path
from varibad_jax.utils.data_utils import DataStager, TRAIN_CHUNK_LEN, TEST_CHUNK_LEN
from varibad_jax.configs.procgen.bigfish.bc import get_config
from varibad_jax.utils.data_utils import (
    merge_trajectories,
    split_data_into_trajectories,
)

if __name__ == "__main__":
    data_dir = Path("/scr/aliang80/varibad_jax/varibad_jax/datasets")
    config = get_config()

    # PROCESS DATA
    for split in ["train", "test"]:
        print("processing split: ", split)
        split_data_dir = (
            data_dir / "procgen" / "expert_data" / config.env.env_id / split
        )

        data = DataStager(
            files=list(split_data_dir.glob("*.npz")),
            chunk_len=TRAIN_CHUNK_LEN if split == "train" else TEST_CHUNK_LEN,
        )

        trajs, max_len = split_data_into_trajectories(data.np_d)
        print(f"max_len: {max_len}")
        num_trajs = len(trajs["observations"])
        print(f"num trajs: {num_trajs}")

        # number of trajectories per chunk
        traj_chunk_size = 500

        chunks_dir = split_data_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        # need to save the trajectories in chunks and compress the data
        for indx, start in enumerate(range(0, num_trajs, traj_chunk_size)):
            chunk = np.arange(start, min(start + traj_chunk_size, num_trajs))
            chunk_file = chunks_dir / f"chunk_{indx}.npz"

            chunk_traj_data = merge_trajectories(
                trajs,
                indices=chunk,
                pad_to=max_len,
            )

            with open(chunk_file, "wb") as f:
                np.savez_compressed(f, **chunk_traj_data)

            print(f"chunk saved to: {chunk_file}")

        print(f"done processing {split} data")
