from absl import logging
import time
import copy
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


def merge_trajectories(data, indices: np.ndarray, pad_to: int = 0):
    if pad_to == 0:
        # just concatenate the trajectories together without padding
        for k, v in data.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    data[k][kk] = np.concatenate([vv[i] for i in indices])
            else:
                data[k] = np.concatenate([v[i] for i in indices])
        return data
    else:
        selected_data = {}
        for k, v in data.items():
            if isinstance(v, dict):
                selected_data[k] = {}
                for kk, vv in v.items():
                    selected_data[k][kk] = [vv[i] for i in indices]
            else:
                selected_data[k] = [v[i] for i in indices]

        # pad everything to the max length
        padded_data = jtu.tree_map(
            lambda x: jnp.pad(
                x,
                ((0, pad_to - x.shape[0]), *[(0, 0)] * (len(x.shape) - 1)),
                mode="constant",
            ),
            selected_data,
        )

        # stack the trajectories together
        stacked_data = {}
        for k, v in padded_data.items():
            if isinstance(v, dict):
                stacked_data[k] = {}
                for kk, vv in v.items():
                    stacked_data[k][kk] = np.stack(vv)
            else:
                stacked_data[k] = np.stack(v)

        return stacked_data


def split_data_into_trajectories(data):
    dones = data["dones"]
    # get indices where the trajectory ends
    traj_ends = np.where(dones)[0]
    traj_ends = np.concatenate([np.array([-1]), traj_ends])
    traj_lens = traj_ends[1:] - traj_ends[:-1]
    max_len = int(traj_lens.max())
    data["mask"] = np.ones_like(data["dones"])
    print("max_len: ", max_len)
    print("num trajs: ", len(traj_lens))

    # split each data by done
    data = jtu.tree_map(lambda x: np.split(x, np.where(dones)[0] + 1)[:-1], data)
    return data, max_len


@chex.dataclass
class Batch:
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray = None
    next_observations: jnp.ndarray = None
    dones: jnp.ndarray = None
    tasks: jnp.ndarray = None
    mask: jnp.ndarray = None
    successes: jnp.ndarray = None
    timestep: jnp.ndarray = None
    traj_index: jnp.ndarray = None
    labelled: jnp.ndarray = None
    latent_actions: jnp.ndarray = None
    is_first: jnp.ndarray = None
    is_last: jnp.ndarray = None
    is_terminal: jnp.ndarray = None
    discount: jnp.ndarray = None


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
        num_workers=8,
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


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    # assert not torch.is_floating_point(obs)
    return obs.astype(np.float64) / 255 - 0.5


class DataStager:
    def __init__(
        self,
        files: list[Path],
        chunk_len: int,
        obs_depth: int = 3,
        seq_len: int = 2,
    ) -> None:

        self.seq_len = seq_len
        self.np_d = None  # type: ignore
        self.obs_depth = obs_depth
        self.files = files
        self.chunk_len = chunk_len
        # random.shuffle(self.files)
        # sort files
        self.files = sorted(self.files)

        num_transitions = self.chunk_len * len(self.files)
        self.np_d = {
            "observations": np.zeros(
                (num_transitions, 64, 64, self.obs_depth), dtype=np.float32
            ),
            "actions": np.zeros(num_transitions, dtype=np.int64),
            "dones": np.zeros(num_transitions, dtype=np.int64),
            "rewards": np.zeros(num_transitions),
            "next_observations": np.zeros(
                (num_transitions, 64, 64, self.obs_depth), dtype=np.float32
            ),
        }

        self._load()

    def _load(self):
        for i, path in tqdm.tqdm(enumerate(self.files)):
            self._load_chunk(path, i)

    def _load_chunk(self, path: Path, i: int):
        keys_map = {
            "observations": "obs",
            "actions": "ta",
            "dones": "done",
            "rewards": "rewards",
            "next_observations": "obs",
        }

        data = np.load(path)
        for k in self.np_d.keys():
            v = data[keys_map[k]]
            if k == "observations" or k == "next_observations":
                v = normalize_obs(v)

            # print(v.shape, self.chunk_len)
            assert len(v) == self.chunk_len, v.shape
            self.np_d[k][i * self.chunk_len : (i + 1) * self.chunk_len] = v

    def get_iter(
        self,
        batch_size: int,
        shuffle=True,
        drop_last=True,
    ) -> Generator[TensorDict, None, None]:
        def collate_fn(batch):
            batch = batch.to_dict()
            return tree_map(lambda x: x.numpy(), batch)

        dataloader = DataLoader(
            self.np_d_unfolded,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
        return dataloader


def load_data(config: ConfigDict, rng: jax.random.PRNGKey):
    start_data_loading = time.time()
    data_dir = Path(config.root_dir) / config.data.data_dir

    # load dataset
    training_env_id = config.env.env_id

    if config.env.env_name == "procgen":
        TRAJS_PER_FILE_TRAIN = 500
        TRAJS_PER_FILE_TEST = 116
        MAX_LEN = 200

        all_data = {}
        for split in ["train", "test"]:
            TRAJS_PER_FILE = (
                TRAJS_PER_FILE_TRAIN if split == "train" else TRAJS_PER_FILE_TEST
            )
            # load chunks
            chunk_dir = (
                data_dir / Path("procgen") / "expert_data" / training_env_id / split
            )
            if config.data.load_latent_actions:
                chunk_dir = chunk_dir / "la_chunks"
            else:
                chunk_dir = chunk_dir / "chunks"

            chunk_files = list(chunk_dir.glob("*.npz"))
            if config.data.debug:
                chunk_files = chunk_files[:1]

            chunk_data = {
                "observations": np.zeros(
                    (TRAJS_PER_FILE * len(chunk_files), MAX_LEN, 64, 64, 3),
                    dtype=np.float32,
                ),
                "actions": np.zeros(
                    (TRAJS_PER_FILE * len(chunk_files), MAX_LEN), dtype=np.int64
                ),
                "rewards": np.zeros(
                    (TRAJS_PER_FILE * len(chunk_files), MAX_LEN), dtype=np.float32
                ),
                "dones": np.zeros(
                    (TRAJS_PER_FILE * len(chunk_files), MAX_LEN), dtype=np.int64
                ),
                "next_observations": np.zeros(
                    (TRAJS_PER_FILE * len(chunk_files), MAX_LEN, 64, 64, 3),
                    dtype=np.float32,
                ),
                "mask": np.zeros(
                    (TRAJS_PER_FILE * len(chunk_files), MAX_LEN), dtype=np.int64
                ),
            }

            if config.data.load_latent_actions:
                chunk_data["latent_actions"] = np.zeros(
                    (
                        TRAJS_PER_FILE * len(chunk_files),
                        MAX_LEN,
                        config.model.latent_action_dim,
                    ),
                    dtype=np.float32,
                )

            for i, chunk_file in tqdm.tqdm(
                enumerate(chunk_files), desc=f"loading {split} files"
            ):
                data = np.load(chunk_file)
                for k, v in data.items():
                    chunk_data[k][i * TRAJS_PER_FILE : (i + 1) * TRAJS_PER_FILE] = v

            all_data[split] = chunk_data
            indices = jnp.arange(TRAJS_PER_FILE * len(chunk_files))
            rng, perm_rng = jax.random.split(rng)

            num_trajs = config.data.num_trajs if split == "train" else 10000
            indices = jax.random.permutation(perm_rng, indices)[:num_trajs]
            all_data[split] = subsample_data(all_data[split], indices)

            # import ipdb

            # ipdb.set_trace()

            if config.data.data_type != "trajectories":

                def apply_mask(x):
                    flatten_first_two = x.reshape(-1, *x.shape[2:])
                    mask = all_data[split]["mask"].reshape(-1) == 1
                    return flatten_first_two[mask]

                all_data[split] = jtu.tree_map(apply_mask, all_data[split])

        return (
            all_data["train"],
            {training_env_id: all_data["test"]},
            {training_env_id: all_data["test"]},
        )

    if config.env.env_name == "procgen":
        dataset_name = Path("procgen") / "expert_data" / training_env_id / "train"
        file = "train_trajs.npz"
    else:
        dataset_name = f"{config.data.dataset_name}_eid-{training_env_id}_n-{config.num_rollouts_collect}"
        file = "traj_dataset.pkl"

    data_path = data_dir / dataset_name / file
    logging.info(f"loading {config.data.data_type} data from {data_path}")

    if config.env.env_name == "procgen":
        data = np.load(data_path, allow_pickle=True)
        data = dict(data)
    else:
        with open(data_path, "rb") as f:
            data = pickle.load(f)

    num_data = data["observations"].shape[0]
    logging.info(f"num data: {num_data}")

    # if we have different eval envs, load the data for that as well
    eval_datasets = {}
    for env_id in config.env.eval_env_ids:
        if env_id != training_env_id:
            if config.env.env_name == "procgen":
                dataset_name = Path("procgen") / "expert_data" / env_id / "test"
                file = "test_trajs.npz"
            else:
                dataset_name = f"{config.data.dataset_name}_eid-{env_id}_n-{config.num_rollouts_collect}"

            eval_dataset_path = data_dir / dataset_name / file

            if not eval_dataset_path.exists():
                logging.info(f"eval dataset not found at {eval_dataset_path}")
                continue

            logging.info(f"loading eval data from {eval_dataset_path}")
            if config.env.env_name == "procgen":
                data = np.load(eval_dataset_path, allow_pickle=True)
                data = dict(data)
            else:
                with open(eval_dataset_path, "rb") as f:
                    eval_data = pickle.load(f)

            if isinstance(eval_data["observations"], list):
                num_eval_data = len(eval_data["observations"])
            else:
                num_eval_data = eval_data["observations"].shape[0]
            logging.info(f"num eval data: {num_eval_data}")

            if "successes" in eval_data:
                del eval_data["successes"]

            eval_datasets[env_id] = eval_data

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

    num_train_trajs = config.data.num_trajs
    if num_train_trajs > -1:
        num_train_trajs = min(num_train_trajs, num_data)
    else:
        num_train_trajs = num_data

    all_indices = jnp.arange(num_data)
    rng, perm_rng = jax.random.split(rng)
    all_indices = jax.random.permutation(perm_rng, all_indices)
    train_indices = all_indices[:num_train_trajs]
    eval_indices = all_indices[num_train_trajs:][:1000]

    train_data = subsample_data(data, train_indices)

    if config.data.add_labelling:
        rng, labelling_rng = jax.random.split(rng)
        # select a couple trajectories to use as trainin
        labelled_indices = jax.random.permutation(labelling_rng, train_indices)[
            : config.data.num_labelled
        ]
        t = train_data["observations"].shape[1]
        labelled = jnp.zeros((num_train_trajs, t))
        labelled = labelled.at[labelled_indices].set(1)
        train_data["labelled"] = labelled

    if config.data.data_type != "trajectories":

        def apply_mask(x):
            flatten_first_two = x.reshape(-1, *x.shape[2:])
            mask = train_data["mask"].reshape(-1) == 1
            return flatten_first_two[mask]

        train_data = jtu.tree_map(apply_mask, train_data)

    # import ipdb

    # ipdb.set_trace()
    eval_data = {}
    eval_data[training_env_id] = subsample_data(data, eval_indices)
    for env_id, env_eval_data in eval_datasets.items():
        num_data = env_eval_data["observations"].shape[0]
        indices = jnp.arange(num_data)
        rng, perm_rng = jax.random.split(rng)
        eval_indices = jax.random.permutation(perm_rng, indices)
        eval_indices = eval_indices[:1000]
        eval_data[env_id] = subsample_data(env_eval_data, eval_indices)

    if config.data.data_type != "trajectories":
        for env_id, env_eval_data in eval_data.items():
            eval_d = eval_data[env_id]

            def apply_mask(x):
                flatten_first_two = x.reshape(-1, *x.shape[2:])
                mask = eval_d["mask"].reshape(-1) == 1
                return flatten_first_two[mask]

            eval_data[env_id] = jtu.tree_map(apply_mask, eval_d)

    prompt_data = {}
    if config.data.data_type == "trajectories":
        prompt_data[training_env_id] = data
        for env_id, env_eval_data in eval_datasets.items():
            prompt_data[env_id] = env_eval_data

    logging.info(f"loading data took: {time.time() - start_data_loading}")
    # logic for creating an unseen set of heldout tasks
    # assuming tasks are the same over timesteps
    # if config.data.holdout_tasks:
    #     if config.env.env_name == "gridworld":
    #         tasks = data["tasks"][:, 0]
    #         unique_tasks = np.unique(tasks, axis=0)

    #         # sample a few to be heldout tasks
    #         indices = rng.choice(range(len(unique_tasks)), size=4, replace=False)
    #         heldout_tasks = unique_tasks[indices]

    #         logging.info(f"heldout tasks: {heldout_tasks}")

    #         eval_mask = (heldout_tasks[:, None] == tasks).all(axis=2).any(axis=0)
    #     elif config.env.env_name == "xland":
    #         # the task is defined by the environment state
    #         # which is the agent's position and the goal position
    #         grid = data["info"]["grid"]
    #         agent_pos = data["info"]["agent_position"]
    #         agent_dir = data["info"]["agent_direction"]

    #         # filter trajectories based on agent starting pos + goal location
    #         ic_mask = (agent_pos[:, 0, 0] < 3) & (agent_pos[:, 0, 1] < 3)
    #         from xminigrid.core.constants import (
    #             TILES_REGISTRY,
    #             Colors,
    #             Tiles,
    #             NUM_COLORS,
    #         )

    #         goal_tile = TILES_REGISTRY[Tiles.BALL, Colors.YELLOW]
    #         goal_locs = jnp.array(
    #             list(zip(*jnp.where((grid[:, 0] == goal_tile).sum(axis=-1))))
    #         )
    #         goal_mask = (goal_locs[:, 1] >= 3) & (goal_locs[:, 2] >= 3)
    #         # eval_mask = ic_mask & goal_mask
    #         eval_mask = goal_mask
    #         logging.info(
    #             f"number of filtered trajectories for evaluation: {eval_mask.sum()}"
    #         )

    #     train_data = subsample_data(data, ~eval_mask)
    #     eval_data = subsample_data(data, eval_mask)

    #     if num_trajs > -1:
    #         dones = data["dones"]
    #         traj = jnp.cumsum(dones)
    #         end_idx = jnp.argmax(traj >= num_trajs) + 1
    #         train_indices = jnp.arange(end_idx)
    #         train_data = subsample_data(train_data, train_indices)
    return train_data, eval_data, prompt_data


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
        data = subsample_data(self.data, index)
        return data

    def __len__(self):
        return self.num_samples


def subsample_window(data, index, window_size):
    fn = lambda x: x[index : index + window_size]
    return tree_map(fn, data)


class TrajectoryDataset(Dataset):
    def __init__(self, data, data_cfg):
        self.data = data
        self.data_cfg = data_cfg
        self.num_trajectories = self.data["observations"].shape[0]

        # set a max length for the trajectories
        self.max_traj_len = (
            self.data["observations"].shape[1] * self.data_cfg.num_trajs_per_batch
        )

        self.context_len = self.data_cfg.context_window
        # import ipdb

        # ipdb.set_trace()

        if "tasks" in self.data:
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
        data["timestep"] = np.zeros(self.max_traj_len, dtype=np.int32)
        data["timestep"][:traj1_len] = np.arange(traj1_len)

        # subsample a context window
        si = np.random.randint(0, max(1, traj1_len - self.context_len))
        data = subsample_window(data, si, self.context_len)

        if self.data_cfg.num_trajs_per_batch > 1:
            for indx in range(self.data_cfg.num_trajs_per_batch - 1):
                if "tasks" in data:
                    task_id = data["tasks"][0]
                    traj_ids = self.task_id_to_traj_map[
                        tuple(task_id)
                    ]  # all the trajectories with the same task

                    # choose a random new trajectory with the same MDP to combine
                    traj_id = np.random.choice(traj_ids)
                else:
                    traj_id = np.random.randint(self.num_trajectories)

                other_traj = subsample_data(self.data, traj_id)
                other_traj["timesteps"] = np.arange(self.max_traj_len, dtype=np.int32)

                si = np.random.randint(
                    0, max(1, other_traj["mask"].sum() - self.context_len)
                )
                other_traj = subsample_window(other_traj, si, self.context_len)

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
                data["timestep"][traj1_len : traj1_len + other_traj_len] = np.arange(
                    other_traj_len
                )

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
        # start = time.time()
        data = subsample_data(self.data, index)
        # logging.info(f"subsample data took: {time.time() - start}")
        return data

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
