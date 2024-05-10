"""
script for collecting offline dataset from trained policy
currently supports only LSTM varibad policy
"""

from absl import app, logging
import os
from pathlib import Path
import pickle
from PIL import Image
import jax
import tqdm
import json
import jax.numpy as jnp
import functools
import haiku as hk
import gymnasium as gym
import jax.tree_util as jtu
from ml_collections import ConfigDict, FieldReference, FrozenConfigDict, config_flags

from varibad_jax.envs.utils import make_envs
from varibad_jax.utils.rollout import run_rollouts
from varibad_jax.models.varibad.varibad import VariBADModel
from varibad_jax.agents.ppo.ppo import PPOAgent
from varibad_jax.models.base import RandomActionAgent

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.01"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_CONFIG = config_flags.DEFINE_config_file("config")


def main(_):
    config = _CONFIG.value
    # first load models from checkpoint
    rng_seq = hk.PRNGSequence(config.seed)

    model_ckpt_dir = Path(config.root_dir) / config.model_ckpt_dir / "model_ckpts"
    ckpt_data = model_ckpt_dir / "best.txt"
    with open(ckpt_data, "r") as f:
        ckpt_data = f.read()
        print(ckpt_data)

    ckpt_config_file = Path(config.root_dir) / config.model_ckpt_dir / "config.json"
    with open(ckpt_config_file, "r") as f:
        config_p = json.load(f)

    config_p = ConfigDict(config_p)
    print(config_p)

    envs, env_params = make_envs(**config_p.env, training=False)
    continuous_actions = not isinstance(envs.action_space, gym.spaces.Discrete)

    if continuous_actions:
        input_action_dim = action_dim = envs.action_space.shape[0]
    else:
        action_dim = envs.action_space.n
        input_action_dim = 1

    agent_model_key = ""

    if "vae" in config_p:
        # load the varibad belief model
        belief_model = VariBADModel(
            config=config_p.vae,
            observation_shape=envs.observation_space.shape,
            action_dim=action_dim,
            input_action_dim=input_action_dim,
            continuous_actions=continuous_actions,
            key=next(rng_seq),
            load_from_ckpt=True,
            ckpt_file=model_ckpt_dir / "best.pkl",
            model_key="belief_model",
        )
        agent_model_key = "agent"
    else:
        belief_model = None

    agent = PPOAgent(
        config=config_p.model,
        observation_shape=envs.observation_space.shape,
        action_dim=action_dim,
        input_action_dim=input_action_dim,
        continuous_actions=continuous_actions,
        key=next(rng_seq),
        load_from_ckpt=True,
        ckpt_file=model_ckpt_dir / "best.pkl",
        model_key=agent_model_key,
    )

    # collect some rollouts
    # steps_per_rollout = config_p.env.num_episodes_per_rollout * envs.max_episode_steps
    steps_per_rollout = (
        config_p.env.num_episodes_per_rollout * config.env.steps_per_rollout
    )

    logging.info("start data collection")
    config_p.num_eval_rollouts = config.num_rollouts_collect

    eval_metrics, transitions, actions = run_rollouts(
        rng=next(rng_seq),
        agent=agent,
        belief_model=belief_model,
        env=envs,
        config=config_p,
        steps_per_rollout=steps_per_rollout,
        action_dim=input_action_dim,
    )

    logging.info(f"eval metrics: {eval_metrics}")

    # save transitions to a dataset format
    data_dir = (
        Path(config.root_dir)
        / "datasets"
        / f"{config.data.dataset_name}_eid-{config.env.env_id}_n-{config.num_rollouts_collect}_steps-{steps_per_rollout}"
    )
    data_dir.mkdir(exist_ok=True, parents=True)
    data_file = data_dir / "dataset.pkl"
    metadata_file = data_dir / "metadata.json"
    dataset_size = actions.shape[0]

    logging.info(f"dataset size: {dataset_size}")

    # render images afterwards
    # all_imgs = []
    # for traj_indx in tqdm.tqdm(range(config.num_rollouts_collect)):
    #     imgs = []
    #     for step in range(config.env.steps_per_rollout):
    #         timestep = jtu.tree_map(lambda x: x[traj_indx][step], transitions)
    #         img = envs.render(envs.env_params, timestep)
    #         img = Image.fromarray(img)
    #         img = img.resize((64, 64))
    #         imgs.append(img)
    #     all_imgs.append(imgs)

    # all_imgs = jnp.array(all_imgs)
    # logging.info(f"all_imgs shape: {all_imgs.shape}")

    # TODO: this is a hack for now

    if config.env.env_name == "xland":
        info = {
            "agent_position": transitions.state.agent.position,
            "agent_direction": transitions.state.agent.direction,
            "grid": transitions.state.grid,
            "goal": transitions.state.goal_encoding,
            "rule": transitions.state.rule_encoding,
        }
        # info = transitions.state
    else:
        info = {}
    #     agent_pos = transitions.state.agent.position  # [B, T, 2]
    #     agent_dir = transitions.state.agent.direction  # [B, T]
    #     observations = jnp.concatenate([agent_pos, agent_dir[:, :, None]], axis=-1)
    #     # observations = transitions.state.grid

    # else:
    observations = transitions.observation
    next_observations = transitions.observation[:, 1:]

    # append the last observation
    next_observations = jnp.concatenate(
        [next_observations, next_observations[:, -2:-1]], axis=1
    )
    rewards = transitions.reward
    dones = transitions.last()

    states = transitions.state

    import ipdb

    ipdb.set_trace()
    if config.env.env_name == "gridworld":
        successes = states.success
        mask = 1 - successes
        mask = mask.at[:, -1].set(0)

        tasks = states.goal
    elif config.env.env_name == "xland":
        mask = transitions.discount

        tasks = states.goal_encoding

    logging.info(
        f"observations shape: {observations.shape}, rewards shape: {rewards.shape}, actions shape: {actions.shape}"
    )

    # actions - [T, B, 1]
    # observations - [T, B, ...]
    data = dict(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        tasks=tasks,
        mask=mask,
        info=info,
        # imgs=all_imgs,
    )

    metadata = {
        "pretrained_model_config": config_p.to_dict(),
        "ckpt_data": ckpt_data,
        "ckpt_dir": str(model_ckpt_dir),
        "avg_ep_return": float(eval_metrics["episode_return"]),
    }
    with open(data_file, "wb") as f:
        pickle.dump(data, f)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    app.run(main)
