from absl import logging
import jax
import chex
import tqdm
from collections import defaultdict as dd
from pathlib import Path
import numpy as np
import pickle
import jax.numpy as jnp
import einops
import tqdm
import numpy.random as npr
from ml_collections.config_dict import ConfigDict
from collections.abc import Generator


@chex.dataclass
class Batch:
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray
    tasks: jnp.ndarray = None
    mask: jnp.ndarray = None


# def load_procgen_data(data_dir: str, env_id: str, stage: str = "train"):
#     data_path = Path(data_dir) / env_id / stage

#     data_files = list(data_path.glob("*.npz"))
#     logging.info(f"num data files: {len(data_files)}")

#     dataset = dd(list)

#     for f in tqdm.tqdm(data_files):
#         data = np.load(f)
#         dataset["observations"].append(data["obs"])
#         dataset["actions"].append(data["ta"])
#         dataset["rewards"].append(data["rewards"])
#         dataset["dones"].append(data["done"])
#         dataset["next_observations"].append(data["obs"])

#     for k in dataset.keys():
#         dataset[k] = np.concatenate(dataset[k], axis=0)
#         logging.info(f"{k}: {dataset[k].shape}")

#     return dataset

import random
import torch
from tensordict import TensorDict, TensorDictBase
from torch.utils.data import DataLoader
from jax.tree_util import tree_map
from torch.utils import data


TRAIN_CHUNK_LEN = 32_768
TEST_CHUNK_LEN = 4096


def numpy_collate(batch):
    import ipdb

    ipdb.set_trace()
    return tree_map(lambda x: x.numpy(), batch)


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def _create_tensordict(length: int, obs_depth) -> TensorDict:
    return TensorDict(
        {
            "obs": torch.zeros(length, obs_depth, 64, 64, dtype=torch.uint8),
            "ta": torch.zeros(length, dtype=torch.long),
            "done": torch.zeros(length, dtype=torch.bool),
            "rewards": torch.zeros(length),
            "ep_returns": torch.zeros(length),
            "values": torch.zeros(length),
        },
        batch_size=length,
        device="cpu",
    )


def _unfold_td(td: TensorDictBase, seq_len: int, unfold_step: int = 1):
    """
    Unfolds the given TensorDict along the time dimension.
    The unfolded TensorDict shares its underlying storage with the original TensorDict.
    """
    res_batch_size = (td.batch_size[0] - seq_len + 1,)
    td = td.apply(
        lambda x: x.unfold(0, seq_len, unfold_step).movedim(-1, 1),
        batch_size=res_batch_size,
    )
    return td


def normalize_obs(obs: torch.Tensor) -> torch.Tensor:
    assert not torch.is_floating_point(obs)
    return obs.float() / 255 - 0.5


class DataStager:
    def __init__(
        self,
        files: list[Path],
        chunk_len: int,
        obs_depth: int = 3,
        seq_len: int = 2,
    ) -> None:

        self.seq_len = seq_len
        self.td: TensorDict = None  # type: ignore
        self.obs_depth = obs_depth
        self.files = files
        self.chunk_len = chunk_len
        random.shuffle(self.files)

        self.td = _create_tensordict(self.chunk_len * len(self.files), self.obs_depth)
        self.td_unfolded = _unfold_td(self.td, self.seq_len, 1)
        self._load()

    def _load(self):
        for i, path in tqdm.tqdm(enumerate(self.files)):
            self._load_chunk(path, i)

    def _load_chunk(self, path: Path, i: int):
        data = np.load(path)
        for k in self.td.keys():
            v = torch.from_numpy(data[k])
            if k == "obs":
                v = v.permute(0, 3, 1, 2)
            assert len(v) == self.chunk_len, v.shape
            self.td[k][i * self.chunk_len : (i + 1) * self.chunk_len] = v

    def get_iter(
        self,
        batch_size: int,
        shuffle=True,
        drop_last=True,
    ) -> Generator[TensorDict, None, None]:
        dataloader = DataLoader(
            self.td_unfolded,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=lambda x: x,
        )
        # dataloader = NumpyLoader(
        #     self.td_unfolded,
        #     batch_size=batch_size,
        #     shuffle=shuffle,
        #     drop_last=drop_last,
        # )
        # data = next(iter(dataloader))

        while True:
            for batch in dataloader:
                batch["obs"] = normalize_obs(batch["obs"])
                batch["obs"] = einops.rearrange(batch["obs"], "B T C H W -> B T H W C")

                numpy_batch = Batch(
                    **{
                        "observations": batch["obs"].numpy(),
                        "actions": batch["ta"].numpy(),
                        "rewards": batch["rewards"].numpy(),
                        "dones": batch["done"].numpy(),
                        "next_observations": batch["obs"].numpy(),
                    }
                )
                yield numpy_batch


def load_data(config: ConfigDict, rng: npr.RandomState, steps_per_rollout: int):
    data_dir = Path(config.root_dir) / config.data_dir

    if config.env.env_name == "procgen":
        train_data = DataStager(
            files=list((data_dir / config.env.env_id / "train").glob("*.npz")),
            chunk_len=TRAIN_CHUNK_LEN,
            seq_len=3,
        )
        eval_data = DataStager(
            files=list((data_dir / config.env.env_id / "test").glob("*.npz")),
            chunk_len=TEST_CHUNK_LEN,
            seq_len=3,
        )
        return train_data, eval_data

    # load dataset
    dataset_name = f"{config.dataset_name}_eid-{config.env.env_id}_n-{config.num_rollouts_collect}_steps-{steps_per_rollout}"
    data_path = data_dir / dataset_name / "dataset.pkl"
    logging.info(f"loading data from {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    num_trajs, num_timesteps = data["observations"].shape[:2]
    avg_return = jnp.mean(jnp.sum(data["rewards"], axis=-1))
    logging.info(f"average return: {avg_return}")

    # TODO: add some observation normalization here
    # data["observations"] /= 14.0

    # need to convert rewards into returns to go
    # do i need to do discounting here?
    if config.model.name == "dt":
        returns = jnp.cumsum(data["rewards"][:, ::-1], axis=1)[:, ::-1]
        data["rewards"] = returns

    # [B, T, *], trajectory data
    if config.num_trajs != -1:
        indices = rng.choice(num_trajs, size=config.num_trajs, replace=False)
        for k, v in data.items():
            data[k] = v[indices]

    for k, v in data.items():
        logging.info(f"{k}: {v.shape}")

    dataset_size = data["observations"].shape[0]
    num_train = int(dataset_size * config.train_frac)

    # split into train and eval
    train_data = {k: v[:num_train] for k, v in data.items()}
    eval_data = {k: v[num_train:] for k, v in data.items()}
    return train_data, eval_data


def create_traj_loader(data, rng, batch_size, num_traj_per_batch: int = 1):
    num_samples = data["observations"].shape[0]
    original_batch_size = batch_size
    batch_size = batch_size * num_traj_per_batch
    num_complete_batches, leftover = divmod(num_samples, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        while True:
            perm = rng.permutation(num_samples)
            for i in range(num_complete_batches):  # only work with complete batches
                # TODO: fix this, a hack to stitch together multiple trajectories
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                batch = Batch(
                    **{
                        k: v[batch_idx].reshape(original_batch_size, -1, *v.shape[2:])
                        for k, v in data.items()
                    }
                )
                yield batch

    batches = data_stream()
    return batches, num_batches


def create_lapo_loader(data, rng, batch_size, num_context: int = 0):
    # sample small windows of transitions in a trajectory
    num_trajs, num_timesteps = data["observations"].shape[:2]
    num_complete_batches, leftover = divmod(num_trajs, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        while True:
            perm = rng.permutation(num_trajs)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                mask = data["mask"][batch_idx]
                valid_ranges = mask.sum(axis=1)

                # sample time_idx that is valid for each trajectory in the batch
                time_idx = rng.randint(valid_ranges)

                batch = Batch(
                    **{
                        k: np.stack(
                            [v[batch_idx, time_idx], v[batch_idx, time_idx + 1]], axis=1
                        )
                        for k, v in data.items()
                    }
                )
                yield batch

    batches = data_stream()
    return batches, num_batches
