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
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags
from typing import Any
import pickle
from pathlib import Path
from ray import train, tune
from ray.train import RunConfig, ScalingConfig
from varibad_jax.train import VAETrainer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_CONFIG = config_flags.DEFINE_config_file("config")

# shorthands for config parameters
psh = {
    "batch_size": "bs",
    "env_id": "eid",
    "hidden_size": "hs",
    "max_episode_steps": "mes",
    "num_epochs": "ne",
    "train_perc": "tp",
    "trainer": "t",
    "num_trajs": "nt",
    "policy_cls": "pc",
    "num_policies": "np",
    "num_eval_episodes": "nee",
}

# run with ray tune
param_space = {}


def train_model_fn(config):
    trial_dir = train.get_context().get_trial_dir()
    if trial_dir:
        print("Trial dir: ", trial_dir)
        config["root_dir"] = Path(trial_dir)
        base_name = Path(trial_dir).name
        config["exp_name"] = base_name
    else:
        suffix = f"{config['exp_name']}_s-{config['seed']}"
        config["root_dir"] = Path(config["root_dir"]) / "results" / suffix

    # wrap config in ConfigDict
    config = ConfigDict(config)

    trainer_cls = VAETrainer
    trainer = trainer_cls(config)
    if config.mode == "train":
        trainer.train()
    elif config.mode == "eval":
        trainer.eval()


def trial_str_creator(trial):
    trial_str = ""
    for k, v in trial.config.items():
        if k in psh and k in param_space:
            trial_str += f"{psh[k]}-{v}_"
    # trial_str += str(trial.trial_id)
    print("trial_str: ", trial_str)
    return trial_str


def main(_):
    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()

    logging.info(f"num_devices: {num_devices}, num_local_devices: {num_local_devices}")

    config = _CONFIG.value.to_dict()
    if config["smoke_test"] is False:
        config.update(param_space)
        train_model = tune.with_resources(train_model_fn, {"cpu": 1, "gpu": 0.2})

        run_config = RunConfig(
            name=config["exp_name"],
            local_dir="/data/anthony/varibad_jax/ray_results",
            storage_path="/data/anthony/varibad_jax/ray_results",
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
