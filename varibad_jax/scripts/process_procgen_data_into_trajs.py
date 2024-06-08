from functools import partial
from absl import logging
import os
import tqdm
import time
import pickle
import json
import h5py
import numpy as np
from pathlib import Path
from varibad_jax.utils.data_utils import DataStager, TRAIN_CHUNK_LEN, TEST_CHUNK_LEN
from varibad_jax.configs.procgen.bigfish.bc import get_config
from varibad_jax.utils.data_utils import (
    merge_trajectories,
    split_data_into_trajectories,
)
import multiprocessing
from multiprocessing import Process

ALL_GAMES = [
    # "bigfish",
    # "bossfight",
    "caveflyer",
    # "chaser",
    # "climber",
    # "coinrun",
    # "dodgeball",
    # "fruitbot",
    # "heist",
    # "jumper",
    # "leaper",
    # "maze",
    # "miner",
    # "ninja",
    # "plunder",
]


def save_trajectory(index_traj_pair, trajs_dir):
    index, traj = index_traj_pair
    npz_file_path = trajs_dir / f"traj_{index}.npz"
    if not os.path.exists(npz_file_path):
        np.savez_compressed(npz_file_path, **traj)


def save_trajectories(trajs, trajs_dir):
    with multiprocessing.Pool() as pool:
        list(
            tqdm.tqdm(
                pool.imap_unordered(
                    partial(save_trajectory, trajs_dir=trajs_dir),
                    enumerate(trajs),
                    chunksize=1,
                ),
                total=len(trajs),
            )
        )


def process_games(games, data_dir, config):
    pid = os.getpid()
    for game in games:
        for split in ["train", "test"]:
            print(f"[{pid}] processing game: {game}, split: {split}")
            split_data_dir = data_dir / "procgen" / "expert_data" / game / split

            data = DataStager(
                files=list(split_data_dir.glob("*.npz"))[:20],
                chunk_len=TRAIN_CHUNK_LEN if split == "train" else TEST_CHUNK_LEN,
            )

            trajs, max_len = split_data_into_trajectories(data.np_d)
            print(f"max_len: {max_len}")
            num_trajs = len(trajs["observations"])
            print(f"num trajs: {num_trajs}")

            # save individual trajectories as npy files
            trajs_dir = split_data_dir / "trajs"
            trajs_dir.mkdir(exist_ok=True)

            # split into lists of dicts
            trajs = [{k: v[i] for k, v in trajs.items()} for i in range(num_trajs)]

            logging.info("saving trajectories")

            start = time.time()
            save_trajectories(trajs, trajs_dir)
            logging.info(
                f"finished processing game: {game}, split: {split}, took: {time.time() - start} seconds"
            )


if __name__ == "__main__":
    from multiprocessing import cpu_count

    data_dir = Path("/scr/aliang80/varibad_jax/varibad_jax/datasets")
    config = get_config()

    for game in ALL_GAMES:
        process_games([game], data_dir, config)
