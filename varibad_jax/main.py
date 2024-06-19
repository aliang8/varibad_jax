from absl import app, logging
import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import os
import tqdm
import pickle
import time
import jax
import flax
import re
import collections
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from typing import Any
import pickle
import jax.tree_util as jtu
from pathlib import Path
from ray import train, tune
from ray.train import RunConfig, ScalingConfig
import tensorflow as tf

from varibad_jax.trainers.meta_trainer import MetaRLTrainer
from varibad_jax.trainers.offline_trainer import OfflineTrainer
from varibad_jax.trainers.rl_trainer import RLTrainer

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
_CONFIG = config_flags.DEFINE_config_file("config")

import warnings

warnings.filterwarnings("ignore")

# shorthands for config parameters
psh = {
    "trainer": "t",
    "batch_size": "bs",
    "seed": "s",
    "env": {
        "env_name": "en",
        "env_id": "eid",
        "eval_env_id": "eval_id",
        "num_frames": "nf",
        "num_processes": "np",
        "steps_per_rollout": "steps",
    },
    "data": {
        "num_trajs": "nt",
        "context_len": "cl",
        "num_transitions": "nt",
        "num_trajs_per_batch": "tpb",
        "add_labelling": "al",
    },
    "vae": {
        "lr": "vlr",
        "kl_weight": "klw",
        "latent_dim": "ld",
        "kl_to_fixed_prior": "klfp",
        "encoder": "enc",
        "num_layers": "nl",
        "num_heads": "nh",
        "dropout_rate": "do",
        "num_vae_updates": "nvu",
        "embedding_dim": "ed",
        "decode_tasks": "dec_t",
        "decode_rewards": "dec_r",
        "decode_states": "dec_s",
    },
    "model": {
        "pass_latent_to_policy": "pltp",
        "pass_task_to_policy": "pttp",
        "name": "pn",
        "idm_nt": "idm_nt",
        "entropy_coeff": "ent_c",
        "idm": {
            "beta": "b",
            "num_codes": "n_codes",
            "code_dim": "code_d",
            "use_state_diff": "usd",
            "idm_scale": "ids",
        },
    },
}

# run with ray tune
param_space = {
    # "model": {"idm_nt": tune.grid_search([20])},
    # "data": {"num_trajs": tune.grid_search([10, 15, 20, 50, 75])},
    "data": {"num_trajs": tune.grid_search([50, 100, 500, 1000])},
    # "data": {"num_trajs": tune.grid_search([10, 20, 50, 100, 500, 1000])},
    "seed": tune.grid_search([0]),
    # "data": {"num_trajs": tune.grid_search([100, 500, 1000, 5000])},
    # "data": {"num_trajs_per_batch": tune.grid_search([2, 3, 4])},
    # "model": {"entropy_coeff": tune.grid_search([0.02, 0.05, 0.1])},
    # "model": {"idm": {"num_codes": tune.grid_search([6, 60, 100])}},
}


def train_model_fn(config):
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.config.experimental.set_visible_devices([], "GPU")

    # tf.config.gpu.set_per_process_memory_growth(True)
    # prevent tf from preallocating all the gpu memory
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    trial_dir = train.get_context().get_trial_dir()
    if trial_dir:
        print("Trial dir: ", trial_dir)
        config["exp_dir"] = trial_dir
        base_name = Path(trial_dir).name
        config["exp_name"] = base_name
        # the group name is without seed
        config["group_name"] = re.sub("_s-\d", "", base_name)
        logging.info(f"wandb group name: {config['group_name']}")
    else:
        exp_name = create_exp_name({}, config)
        config["exp_dir"] = str(Path(config["exp_dir"]) / "results" / exp_name)
        config["exp_name"] = exp_name

    # wrap config in ConfigDict
    config = ConfigDict(config)

    if config.trainer == "rl":
        trainer_cls = RLTrainer
    elif config.trainer == "vae":
        trainer_cls = MetaRLTrainer
    elif config.trainer == "offline":
        trainer_cls = OfflineTrainer
    else:
        raise ValueError(f"trainer {config.trainer} not found")

    trainer = trainer_cls(config)
    if config.mode == "train":
        trainer.train()
    elif config.mode == "eval":
        trainer.eval(epoch=0)


def update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if type(source[key]) != type(overrides[key]):
            source[key] = overrides[key]
        elif isinstance(value, collections.abc.Mapping) and value:
            returned = update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def create_exp_name(param_space, config):
    global trial_str
    global keys_added
    trial_str = config["exp_name"] + ","
    keys_added = set()

    def add_to_trial_str(k, v):
        global trial_str
        global keys_added

        shorthand = psh
        value = config
        for i, key in enumerate(k):
            key_str = eval(str(key))[0]
            if key_str == "grid_search" or isinstance(key, jtu.SequenceKey):
                continue

            shorthand = shorthand[key_str]
            value = value[key_str]

        if not shorthand in keys_added:
            trial_str += str(shorthand) + "-" + str(value) + ","

        keys_added.add(shorthand)

    jtu.tree_map_with_path(add_to_trial_str, param_space)
    jtu.tree_map_with_path(add_to_trial_str, config["keys_to_include"])

    trial_str = trial_str[:-1]
    # print("trial_str: ", trial_str)
    return trial_str


def trial_str_creator(trial):
    return create_exp_name(param_space, trial.config)


def main(_):
    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()

    logging.info(f"num_devices: {num_devices}, num_local_devices: {num_local_devices}")

    config = _CONFIG.value.to_dict()

    if config["enable_jit"] is False:
        jax.config.update("jax_disable_jit", True)

    if config["smoke_test"] is False:
        config["use_wb"] = True  # log to wandb
        # update dict of dict
        config = update(config, param_space)
        train_model = tune.with_resources(
            train_model_fn, {"cpu": config["cpu"], "gpu": config["gpu"]}
        )

        ray_path = Path(config["root_dir"]) / config["ray_logdir"]
        run_config = RunConfig(
            name=config["exp_name"],
            local_dir=ray_path,
            storage_path=ray_path,
            log_to_file=True,
        )
        tuner = tune.Tuner(
            train_model,
            param_space=config,
            run_config=run_config,
            tune_config=tune.TuneConfig(
                trial_name_creator=trial_str_creator,
                trial_dirname_creator=trial_str_creator,
            ),
        )
        results = tuner.fit()
        print(results)
    else:
        # run without ray tune
        train_model_fn(config)


if __name__ == "__main__":
    app.run(main)
