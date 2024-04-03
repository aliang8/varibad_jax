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
import collections
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from typing import Any
import pickle
from pathlib import Path
from ray import train, tune
from ray.train import RunConfig, ScalingConfig
from varibad_jax.trainers.meta_trainer import VAETrainer
from varibad_jax.trainers.rl_trainer import RLTrainer
from varibad_jax.trainers.offline_trainer import OfflineTrainer

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_CONFIG = config_flags.DEFINE_config_file("config")

# shorthands for config parameters
psh = {
    "batch_size": "bs",
    "seed": "s",
    "env": {
        "env_id": "eid",
        "num_frames": "nf",
        "num_processes": "np",
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
    },
    "policy": {
        "pass_latent_to_policy": "pltp",
        "pass_task_to_policy": "pttp",
        "name": "pn",
    },
}

# run with ray tune

# params for varibad vae exps
param_space = {
    "seed": tune.grid_search([0]),
}

# params for offline RL DT exps
param_space = {
    "seed": tune.grid_search([0]),
}


def train_model_fn(config):
    trial_dir = train.get_context().get_trial_dir()
    if trial_dir:
        print("Trial dir: ", trial_dir)
        config["exp_dir"] = Path(trial_dir)
        base_name = Path(trial_dir).name
        config["exp_name"] = base_name
    else:
        suffix = f"{config['exp_name']}_s-{config['seed']}"
        config["exp_dir"] = Path(config["exp_dir"]) / "results" / suffix

    # wrap config in ConfigDict
    config = ConfigDict(config)

    if config.trainer == "rl":
        trainer_cls = RLTrainer
    elif config.trainer == "vae":
        trainer_cls = VAETrainer
    elif config.trainer == "offline":
        trainer_cls = OfflineTrainer

    trainer = trainer_cls(config)
    if config.mode == "train":
        trainer.train()
    elif config.mode == "eval":
        trainer.eval()


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


def trial_str_creator(trial):
    trial_str = ""

    for k, override in param_space.items():
        if k in trial.config:
            if isinstance(override, dict) and "grid_search" not in override:
                for k2 in override.keys():
                    if k2 in trial.config[k]:
                        trial_str += f"{psh[k][k2]}-{trial.config[k][k2]}_"
            else:
                trial_str += f"{psh[k]}-{trial.config[k]}_"

    # also add keys to include
    for k, v in trial.config["keys_to_include"].items():
        for k2 in v:
            trial_str += f"{k[0]}-{trial.config[k][k2]}_"

    trial_str = trial_str[:-1]
    print("trial_str: ", trial_str)
    return trial_str


def main(_):
    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()

    logging.info(f"num_devices: {num_devices}, num_local_devices: {num_local_devices}")

    config = _CONFIG.value.to_dict()
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
