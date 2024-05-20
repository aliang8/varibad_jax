from absl import logging
import math
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
import jax.tree_util as jtu
from ml_collections.config_dict import ConfigDict
from collections.abc import Generator
from torch.utils.data import Dataset, DataLoader


@chex.dataclass
class Batch:
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray
    tasks: jnp.ndarray = None
    mask: jnp.ndarray = None
    successes: jnp.ndarray = None
    traj_index: jnp.ndarray = None


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
from collections import defaultdict as dd

TRAIN_CHUNK_LEN = 32_768
TEST_CHUNK_LEN = 4096


def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))


def subsample_data(data, indices):
    fn = lambda x: x[indices]
    subsample_data = tree_map(fn, data)
    return subsample_data


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
        num_transitions: int = -1,
    ) -> None:

        self.seq_len = seq_len
        self.td: TensorDict = None  # type: ignore
        self.obs_depth = obs_depth
        self.files = files
        self.chunk_len = chunk_len
        random.shuffle(self.files)

        self.td = _create_tensordict(self.chunk_len * len(self.files), self.obs_depth)
        self.td_unfolded = _unfold_td(self.td, self.seq_len, 1)

        if num_transitions != -1:
            self.td_unfolded = self.td_unfolded[:num_transitions]

        logging.info(f"num transitions: {len(self.td_unfolded['obs'])}")

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


def convert_to_trajectories(data):
    dones = data["dones"]
    # get indices where the trajectory ends
    traj_ends = jnp.where(dones)[0]
    traj_ends = jnp.concatenate([jnp.array([-1]), traj_ends])
    traj_lens = traj_ends[1:] - traj_ends[:-1]
    max_len = int(traj_lens.max())

    data["mask"] = jnp.ones_like(data["dones"])

    # split each data by done
    data = tree_map(lambda x: jnp.split(x, jnp.where(dones)[0] + 1)[:-1], data)

    if "successes" in data:
        del data["successes"]

    # pad everything to the max length
    data = tree_map(
        lambda x: jnp.pad(
            x,
            ((0, max_len - x.shape[0]), *[(0, 0)] * (len(x.shape) - 1)),
            mode="constant",
        ),
        data,
    )
    # stack the trajectories together
    stacked_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            stacked_data[k] = {}
            for kk, vv in v.items():
                stacked_data[k][kk] = jnp.stack(vv)
        else:
            stacked_data[k] = jnp.stack(v)

    data = stacked_data
    return data


def load_data(config: ConfigDict, rng: npr.RandomState, steps_per_rollout: int):
    data_dir = Path(config.root_dir) / config.data.data_dir

    if config.env.env_name == "procgen":
        train_data = DataStager(
            files=list(
                (
                    data_dir / "procgen" / "expert_data" / config.env.env_id / "train"
                ).glob("*.npz")
            ),
            chunk_len=TRAIN_CHUNK_LEN,
            seq_len=2 + config.data.context_len,
            num_transitions=config.data.num_transitions,
        )
        eval_data = DataStager(
            files=list(
                (
                    data_dir / "procgen" / "expert_data" / config.env.env_id / "test"
                ).glob("*.npz")
            ),
            chunk_len=TEST_CHUNK_LEN,
            seq_len=2 + config.data.context_len,
        )
        return train_data, eval_data

    # load dataset
    dataset_name = f"{config.data.dataset_name}_eid-{config.env.env_id}_n-{config.num_rollouts_collect}"
    data_path = data_dir / dataset_name / "dataset.pkl"
    logging.info(f"loading data from {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    num_data = data["observations"].shape[0]
    logging.info(f"num data: {num_data}")

    # also load eval dataset
    if config.env.env_id != config.env.eval_env_id:
        dataset_name = f"{config.data.dataset_name}_eid-{config.env.eval_env_id}_n-500"
        eval_dataset_path = data_dir / dataset_name / "dataset.pkl"
        logging.info(f"loading eval data from {eval_dataset_path}")
        with open(eval_dataset_path, "rb") as f:
            eval_data = pickle.load(f)
        num_eval_data = eval_data["observations"].shape[0]
        logging.info(f"num eval data: {num_eval_data}")

        # combine the two datasets
        data = jtu.tree_map(
            lambda x, y: np.concatenate([x, y], axis=0), data, eval_data
        )

        if "successes" in eval_data:
            del eval_data["successes"]

    if "successes" in data:
        del data["successes"]

    if config.env.env_name == "xland" and not config.env.symbolic_obs:
        data["observations"] = data["imgs"]
        del data["imgs"]
        # normalize the imgs
        data["observations"] = (data["observations"].astype(np.float32) / 255.0) - 0.5

    # num_trajs, num_timesteps = data["observations"].shape[:2]
    # avg_return = jnp.mean(jnp.sum(data["rewards"], axis=-1))
    # logging.info(f"average return: {avg_return}")

    # TODO: add some observation normalization here
    # data["observations"] /= 14.0

    # need to convert rewards into returns to go
    # do i need to do discounting here?
    if config.model.name == "dt":
        returns = jnp.cumsum(data["rewards"][:, ::-1], axis=1)[:, ::-1]
        data["rewards"] = returns

    fn = lambda x, y: logging.info(f"{jax.tree_util.keystr(x)}: {y.shape}")
    jtu.tree_map_with_path(fn, data)

    # logic for creating an unseen set of heldout tasks
    # assuming tasks are the same over timesteps
    if config.data.holdout_tasks:
        if config.env.env_name == "gridworld":
            tasks = data["tasks"][:, 0]
            unique_tasks = np.unique(tasks, axis=0)

            # sample a few to be heldout tasks
            indices = rng.choice(range(len(unique_tasks)), size=4, replace=False)
            heldout_tasks = unique_tasks[indices]

            logging.info(f"heldout tasks: {heldout_tasks}")

            eval_mask = (heldout_tasks[:, None] == tasks).all(axis=2).any(axis=0)
        elif config.env.env_name == "xland":
            # the task is defined by the environment state
            # which is the agent's position and the goal position
            grid = data["info"]["grid"]
            agent_pos = data["info"]["agent_position"]
            agent_dir = data["info"]["agent_direction"]

            # filter trajectories based on agent starting pos + goal location
            ic_mask = (agent_pos[:, 0, 0] < 3) & (agent_pos[:, 0, 1] < 3)
            from xminigrid.core.constants import (
                TILES_REGISTRY,
                Colors,
                Tiles,
                NUM_COLORS,
            )

            goal_tile = TILES_REGISTRY[Tiles.BALL, Colors.YELLOW]
            goal_locs = jnp.array(
                list(zip(*jnp.where((grid[:, 0] == goal_tile).sum(axis=-1))))
            )
            goal_mask = (goal_locs[:, 1] >= 3) & (goal_locs[:, 2] >= 3)
            # eval_mask = ic_mask & goal_mask
            eval_mask = goal_mask
            logging.info(
                f"number of filtered trajectories for evaluation: {eval_mask.sum()}"
            )

        train_data = subsample_data(data, ~eval_mask)
        eval_data = subsample_data(data, eval_mask)

        if num_trajs > -1:
            dones = data["dones"]
            traj = jnp.cumsum(dones)
            end_idx = jnp.argmax(traj >= num_trajs) + 1
            train_indices = jnp.arange(end_idx)
            train_data = subsample_data(train_data, train_indices)
    else:
        dataset_size = data["observations"].shape[0]
        num_total_trajs = data["dones"].sum()
        num_trajs = config.data.num_trajs
        if num_trajs > -1:
            num_trajs = min(num_trajs, num_total_trajs)
        else:
            num_trajs = num_total_trajs

        num_train_trajs = math.ceil(num_trajs * config.data.train_frac)
        dones = data["dones"]
        traj = jnp.cumsum(dones)
        train_end_idx = jnp.argmax(traj >= num_train_trajs) + 1
        end_idx = jnp.argmax(traj >= num_trajs) + 1

        train_indices = jnp.arange(end_idx)[:train_end_idx]
        eval_indices = jnp.arange(end_idx)[train_end_idx:]

        # split into train and eval
        train_data = subsample_data(data, train_indices)

        if config.env.env_id == config.env.eval_env_id:
            eval_data = subsample_data(data, eval_indices)

    if config.data.data_type == "trajectories":
        logging.info("converting data to trajectories")
        train_data = convert_to_trajectories(train_data)
        if eval_data["observations"].shape[0] > 0:
            eval_data = convert_to_trajectories(eval_data)

    logging.info(f"num training data: {train_data['observations'].shape[0]}")
    logging.info(f"num eval data: {eval_data['observations'].shape[0]}")

    return train_data, eval_data


class TransitionsDataset(Dataset):
    def __init__(self, data, data_cfg):
        self.data = data
        self.data_cfg = data_cfg

        # # subsample where only the o_t is valid
        # mask = self.data["mask"][:, -2]
        # for k, v in self.data.items():
        #     self.data[k] = v[torch.where(mask > 0)]

        self.num_samples = self.data["observations"].shape[0]

    def __getitem__(self, index):
        return subsample_data(self.data, index)

    def __len__(self):
        return self.num_samples


class TrajectoryDataset(Dataset):
    def __init__(self, data, data_cfg):
        self.data = data
        self.data_cfg = data_cfg
        self.num_trajectories = self.data["observations"].shape[0]

        # set a max length for the trajectories
        self.max_traj_len = (
            self.data["observations"].shape[1] * self.data_cfg.num_trajs_per_batch
        )

        # map from trajectory index to task index
        self.tasks = self.data["tasks"][:, 0]
        self.traj_to_task_id_map = {}
        for indx, task in enumerate(self.tasks):
            self.traj_to_task_id_map[indx] = task

        self.task_id_to_traj_map = {}
        for indx, task in enumerate(self.tasks):
            if tuple(task) not in self.task_id_to_traj_map:
                self.task_id_to_traj_map[tuple(task)] = []
            self.task_id_to_traj_map[tuple(task)].append(indx)

    def __getitem__(self, index):
        data = subsample_data(self.data, index)

        data["traj_index"] = np.zeros(self.max_traj_len, dtype=np.int32)
        traj1_len = data["mask"].sum()
        data["traj_index"][:traj1_len] = 0

        if self.data_cfg.num_trajs_per_batch > 1:
            for indx in range(self.data_cfg.num_trajs_per_batch - 1):
                task_id = data["tasks"][0]
                traj_ids = self.task_id_to_traj_map[
                    tuple(task_id)
                ]  # all the trajectories with the same task

                # choose a random new trajectory with the same MDP to combine
                traj_id = np.random.choice(traj_ids)
                other_traj = subsample_data(self.data, traj_id)

                # compute total length of the trajectories
                traj1_len = int(data["mask"].sum())
                other_traj_len = int(other_traj["mask"].sum())

                # merge the two trajectories together
                for k, v in other_traj.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            new_vv = np.zeros((self.max_traj_len, *vv.shape[1:]))
                            # print(k, kk, data[k][kk].shape)
                            new_vv[:traj1_len] = data[k][kk][:traj1_len]
                            new_vv[traj1_len : traj1_len + other_traj_len] = other_traj[
                                k
                            ][kk][:other_traj_len]
                            data[k][kk] = new_vv
                            # print(k, kk, data[k][kk].shape)
                    else:
                        new_v = np.zeros((self.max_traj_len, *v.shape[1:]))
                        new_v[:traj1_len] = data[k][:traj1_len]
                        new_v[traj1_len : traj1_len + other_traj_len] = other_traj[k][
                            :other_traj_len
                        ]
                        data[k] = new_v
                        # print(k, data[k].shape)

                data["traj_index"][traj1_len : traj1_len + other_traj_len] = indx + 1

        fn = lambda x: torch.Tensor(x)
        data = tree_map(fn, data)
        return data

    def __len__(self):
        return self.num_trajectories


class LAPODataset(Dataset):
    def __init__(self, data, data_cfg):
        self.data = data
        self.data_cfg = data_cfg

        # not sure what to do with this stuff
        if "info" in self.data:
            del self.data["info"]

        # restructure it to be N-step transitions
        for k, v in self.data.items():
            v = torch.from_numpy(v)

            # unfold it so we can get N-step transitions
            v = v.unfold(0, self.data_cfg.context_len + 2, 1).movedim(-1, 1)
            self.data[k] = v

        # filter trajectories where the any of the previous observations are done
        mask = self.data["dones"][:, :-1].any(axis=-1)
        for k, v in self.data.items():
            self.data[k] = v[torch.where(~mask)]

        self.num_samples = self.data["observations"].shape[0]

    def __getitem__(self, index):
        return subsample_data(self.data, index)

    def __len__(self):
        return self.num_samples


def create_data_loader(data, data_cfg):
    fn = lambda x: np.asarray(x)
    data = tree_map(fn, data)

    if data_cfg.data_type == "trajectories":
        torch_dataset = TrajectoryDataset(data, data_cfg)
    elif data_cfg.data_type == "lapo":
        torch_dataset = LAPODataset(data, data_cfg)
    elif data_cfg.data_type == "transitions":
        torch_dataset = TransitionsDataset(data, data_cfg)

    logging.info(f"num samples: {len(torch_dataset)}")

    return NumpyLoader(
        torch_dataset,
        batch_size=min(len(torch_dataset), data_cfg.batch_size),
        shuffle=True,
        drop_last=False,
    )
