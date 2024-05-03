from absl import logging
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import pyrallis
import wandb
import xminigrid
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState
from varibad_jax.varibad_jax.scripts.utils import (
    Transition,
    calculate_gae,
    ppo_update_networks,
    rollout,
)
from xminigrid.environment import Environment, EnvParams
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.benchmarks import Benchmark, load_benchmark, load_benchmark_from_path

import math
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import glorot_normal, orthogonal, zeros_init
from typing import TypedDict

# this will be default in new jax versions anyway
# jax.config.update("jax_threefry_partitionable", True)


class GRU(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, xs, init_state):
        seq_len, input_dim = xs.shape
        # this init might not be optimal, for example bias for reset gate should be -1 (for now ok)
        Wi = self.param(
            "Wi", glorot_normal(in_axis=1, out_axis=0), (self.hidden_dim * 3, input_dim)
        )
        Wh = self.param(
            "Wh", orthogonal(column_axis=0), (self.hidden_dim * 3, self.hidden_dim)
        )
        bi = self.param("bi", zeros_init(), (self.hidden_dim * 3,))
        bn = self.param("bn", zeros_init(), (self.hidden_dim,))

        def _step_fn(h, x):
            igates = jnp.split(Wi @ x + bi, 3)
            hgates = jnp.split(Wh @ h, 3)

            reset = nn.sigmoid(igates[0] + hgates[0])
            update = nn.sigmoid(igates[1] + hgates[1])
            new = nn.tanh(igates[2] + reset * (hgates[2] + bn))
            next_h = (1 - update) * new + update * h

            return next_h, next_h

        last_state, all_states = jax.lax.scan(_step_fn, init=init_state, xs=xs)
        return all_states, last_state


class RNNModel(nn.Module):
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, xs, init_state):
        # xs: [seq_len, input_dim]
        # init_state: [num_layers, hidden_dim]
        outs, states = [], []
        for layer in range(self.num_layers):
            xs, state = GRU(hidden_dim=self.hidden_dim)(xs, init_state[layer])
            outs.append(xs)
            states.append(state)

        # sum outputs from all layers, kinda like in ResNet
        return jnp.array(outs).sum(0), jnp.array(states)


BatchedRNNModel = flax.linen.vmap(
    RNNModel,
    variable_axes={"params": None},
    split_rngs={"params": False},
    axis_name="batch",
)


class ActorCriticInput(TypedDict):
    observation: jax.Array
    prev_action: jax.Array
    prev_reward: jax.Array


class ActorCriticRNN(nn.Module):
    num_actions: int
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 64
    rnn_num_layers: int = 1
    head_hidden_dim: int = 64
    img_obs: bool = False

    @nn.compact
    def __call__(
        self, inputs: ActorCriticInput, hidden: jax.Array
    ) -> tuple[distrax.Categorical, jax.Array, jax.Array]:
        B, S = inputs["observation"].shape[:2]
        # encoder from https://github.com/lcswillems/rl-starter-files/blob/master/model.py
        if self.img_obs:
            img_encoder = nn.Sequential(
                [
                    nn.Conv(
                        16,
                        (3, 3),
                        strides=2,
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                    ),
                    nn.relu,
                    nn.Conv(
                        32,
                        (3, 3),
                        strides=2,
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                    ),
                    nn.relu,
                    nn.Conv(
                        32,
                        (3, 3),
                        strides=2,
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                    ),
                    nn.relu,
                    nn.Conv(
                        32,
                        (3, 3),
                        strides=2,
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                    ),
                ]
            )
        else:
            img_encoder = nn.Sequential(
                [
                    nn.Conv(
                        16,
                        (2, 2),
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                    ),
                    nn.relu,
                    # use this only for image sizes >= 7
                    # MaxPool2d((2, 2)),
                    nn.Conv(
                        32,
                        (2, 2),
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                    ),
                    nn.relu,
                    nn.Conv(
                        64,
                        (2, 2),
                        padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)),
                    ),
                    nn.relu,
                ]
            )
        action_encoder = nn.Embed(self.num_actions, self.action_emb_dim)

        rnn_core = BatchedRNNModel(self.rnn_hidden_dim, self.rnn_num_layers)
        actor = nn.Sequential(
            [
                nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(2)),
                nn.tanh,
                nn.Dense(self.num_actions, kernel_init=orthogonal(0.01)),
            ]
        )
        critic = nn.Sequential(
            [
                nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(2)),
                nn.tanh,
                nn.Dense(1, kernel_init=orthogonal(1.0)),
            ]
        )

        # [batch_size, seq_len, ...]
        obs_emb = img_encoder(inputs["observation"]).reshape(B, S, -1)
        act_emb = action_encoder(inputs["prev_action"])
        # [batch_size, seq_len, hidden_dim + act_emb_dim + 1]
        out = jnp.concatenate(
            [obs_emb, act_emb, inputs["prev_reward"][..., None]], axis=-1
        )
        # core networks
        out, new_hidden = rnn_core(out, hidden)
        dist = distrax.Categorical(logits=actor(out))
        values = critic(out)

        return dist, jnp.squeeze(values, axis=-1), new_hidden

    def initialize_carry(self, batch_size):
        return jnp.zeros((batch_size, self.rnn_num_layers, self.rnn_hidden_dim))


@dataclass
class TrainConfig:
    project: str = "xminigrid"
    group: str = "glamor"
    name: str = "single-task-ppo"
    env_id: str = "XLand-MiniGrid-R1-9x9"
    benchmark_id: Optional[str] = None
    ruleset_id: Optional[int] = 1
    img_obs: bool = False
    # agent
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 1024
    rnn_num_layers: int = 1
    head_hidden_dim: int = 256
    # training
    num_envs: int = 8192
    num_steps: int = 16
    update_epochs: int = 1
    num_minibatches: int = 16
    total_timesteps: int = 5_000_000
    lr: float = 0.001
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_episodes: int = 80
    seed: int = 42

    def __post_init__(self):
        num_devices = jax.local_device_count()
        # num_devices = 1
        # splitting computation across all available devices
        self.num_envs_per_device = self.num_envs // num_devices
        self.total_timesteps_per_device = self.total_timesteps // num_devices
        self.eval_episodes_per_device = self.eval_episodes // num_devices
        print(
            f"Num devices: {num_devices}, Num envs per device: {self.num_envs_per_device}"
        )
        assert self.num_envs % num_devices == 0
        self.num_updates = (
            self.total_timesteps_per_device
            // self.num_steps
            // self.num_envs_per_device
        )
        print(f"Num devices: {num_devices}, Num updates: {self.num_updates}")


def make_states(config: TrainConfig):
    # for learning rate scheduling
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.num_minibatches * config.update_epochs))
            / config.num_updates
        )
        return config.lr * frac

    # setup environment
    env, env_params = xminigrid.make(config.env_id)
    env = GymAutoResetWrapper(env)

    # for single-task XLand environments
    # if config.benchmark_id is not None:
    #     assert (
    #         "XLand-MiniGrid" in config.env_id
    #     ), "Benchmarks should be used only with XLand environments."
    #     assert (
    #         config.ruleset_id is not None
    #     ), "Ruleset ID should be specified for benchmarks usage."
    # benchmark = xminigrid.load_benchmark(config.benchmark_id)
    benchmark_path = (
        "/home/anthony/varibad_jax/varibad_jax/envs/xland_benchmarks/test_ruleset.pkl"
    )
    benchmark = load_benchmark_from_path(benchmark_path)
    env_params = env_params.replace(ruleset=benchmark.get_ruleset(config.ruleset_id))

    # enabling image observations if needed
    if config.img_obs:
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper

        env = RGBImgObservationWrapper(env)

    # setup training state
    rng = jax.random.PRNGKey(config.seed)
    rng, _rng = jax.random.split(rng)

    network = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        action_emb_dim=config.action_emb_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        head_hidden_dim=config.head_hidden_dim,
        img_obs=config.img_obs,
    )
    # [batch_size, seq_len, ...]
    init_obs = {
        "observation": jnp.zeros(
            (config.num_envs_per_device, 1, *env.observation_shape(env_params))
        ),
        "prev_action": jnp.zeros((config.num_envs_per_device, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((config.num_envs_per_device, 1)),
    }
    init_hstate = network.initialize_carry(batch_size=config.num_envs_per_device)

    network_params = network.init(_rng, init_obs, init_hstate)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(
            learning_rate=linear_schedule, eps=1e-8
        ),  # eps=1e-5
    )
    train_state = TrainState.create(
        apply_fn=network.apply, params=network_params, tx=tx
    )

    return rng, env, env_params, init_hstate, train_state


def make_train(
    env: Environment,
    env_params: EnvParams,
    config: TrainConfig,
    wandb_run=None,
):
    @partial(jax.pmap, axis_name="devices")
    # @jax.jit
    def train(
        rng: jax.Array,
        train_state: TrainState,
        init_hstate: jax.Array,
    ):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs_per_device)

        timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_rng)
        prev_action = jnp.zeros(config.num_envs_per_device, dtype=jnp.int32)
        prev_reward = jnp.zeros(config.num_envs_per_device)

        # TRAIN LOOP
        def _update_step(runner_state, _):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                (
                    rng,
                    train_state,
                    prev_timestep,
                    prev_action,
                    prev_reward,
                    prev_hstate,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                dist, value, hstate = train_state.apply_fn(
                    train_state.params,
                    {
                        # [batch_size, seq_len=1, ...]
                        "observation": prev_timestep.observation[:, None],
                        "prev_action": prev_action[:, None],
                        "prev_reward": prev_reward[:, None],
                    },
                    prev_hstate,
                )
                action, log_prob = dist.sample_and_log_prob(seed=_rng)
                # squeeze seq_len where possible
                action, value, log_prob = (
                    action.squeeze(1),
                    value.squeeze(1),
                    log_prob.squeeze(1),
                )

                # STEP ENV
                timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(
                    env_params, prev_timestep, action
                )
                transition = Transition(
                    done=timestep.last(),
                    action=action,
                    value=value,
                    reward=timestep.reward,
                    log_prob=log_prob,
                    obs=prev_timestep.observation,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                )
                runner_state = (
                    rng,
                    train_state,
                    timestep,
                    action,
                    timestep.reward,
                    hstate,
                )
                return runner_state, transition

            initial_hstate = runner_state[-1]
            # transitions: [seq_len, batch_size, ...]
            runner_state, transitions = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            rng, train_state, timestep, prev_action, prev_reward, hstate = runner_state
            # calculate value of the last step for bootstrapping
            _, last_val, _ = train_state.apply_fn(
                train_state.params,
                {
                    "observation": timestep.observation[:, None],
                    "prev_action": prev_action[:, None],
                    "prev_reward": prev_reward[:, None],
                },
                hstate,
            )
            advantages, targets = calculate_gae(
                transitions, last_val.squeeze(1), config.gamma, config.gae_lambda
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, transitions, advantages, targets = batch_info
                    new_train_state, update_info = ppo_update_networks(
                        train_state=train_state,
                        transitions=transitions,
                        init_hstate=init_hstate.squeeze(1),
                        advantages=advantages,
                        targets=targets,
                        clip_eps=config.clip_eps,
                        vf_coef=config.vf_coef,
                        ent_coef=config.ent_coef,
                    )
                    return new_train_state, update_info

                rng, train_state, init_hstate, transitions, advantages, targets = (
                    update_state
                )
                # print("HERE")
                # jax.debug.print("rng: {}", rng)

                # MINIBATCHES PREPARATION
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config.num_envs_per_device)
                # [seq_len, batch_size, ...]
                batch = (init_hstate, transitions, advantages, targets)
                # [batch_size, seq_len, ...], as our model assumes
                batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                shuffled_batch = jtu.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # [num_minibatches, minibatch_size, ...]
                minibatches = jtu.tree_map(
                    lambda x: jnp.reshape(
                        x, (config.num_minibatches, -1) + x.shape[1:]
                    ),
                    shuffled_batch,
                )
                train_state, update_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                update_state = (
                    rng,
                    train_state,
                    init_hstate,
                    transitions,
                    advantages,
                    targets,
                )
                return update_state, update_info

            # [seq_len, batch_size, num_layers, hidden_dim]
            init_hstate = initial_hstate[None, :]
            update_state = (
                rng,
                train_state,
                init_hstate,
                transitions,
                advantages,
                targets,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )

            # averaging over minibatches then over epochs
            loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

            rng, train_state = update_state[:2]
            # EVALUATE AGENT
            rng, _rng = jax.random.split(rng)
            eval_rng = jax.random.split(_rng, num=config.eval_episodes_per_device)

            # vmap only on rngs
            eval_stats = jax.vmap(rollout, in_axes=(0, None, None, None, None, None))(
                eval_rng,
                env,
                env_params,
                train_state,
                # TODO: make this as a static method mb?
                jnp.zeros((1, config.rnn_num_layers, config.rnn_hidden_dim)),
                1,
            )
            eval_stats = jax.lax.pmean(eval_stats, axis_name="devices")
            loss_info.update(
                {
                    "eval/returns": eval_stats.reward.mean(0),
                    "eval/lengths": eval_stats.length.mean(0),
                    "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
                }
            )
            runner_state = (
                rng,
                train_state,
                timestep,
                prev_action,
                prev_reward,
                hstate,
            )
            return runner_state, loss_info

        runner_state = (
            rng,
            train_state,
            timestep,
            prev_action,
            prev_reward,
            init_hstate,
        )
        runner_state, loss_info = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )
        return {"runner_state": runner_state, "loss_info": loss_info}

    return train


@pyrallis.wrap()
def train(config: TrainConfig):
    # logging to wandb
    # run = wandb.init(
    #     project=config.project,
    #     group=config.group,
    #     name=config.name,
    #     config=asdict(config),
    #     save_code=True,
    # )

    rng, env, env_params, init_hstate, train_state = make_states(config)
    # replicating args across devices
    rng = jax.random.split(rng, num=jax.local_device_count())
    train_state = replicate(train_state, jax.local_devices())
    init_hstate = replicate(init_hstate, jax.local_devices())

    # print("Compiling...")
    t = time.time()
    train_fn = make_train(env, env_params, config, None)
    # train_fn = train_fn.lower(rng, train_state, init_hstate).compile()
    # elapsed_time = time.time() - t
    # print(f"Done in {elapsed_time:.2f}s.")

    print("Training...")
    # t = time.time()
    # train_info = jax.block_until_ready(train_fn(rng, train_state, init_hstate))

    train_info = train_fn(rng, train_state, init_hstate)
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s")

    print("Logging...")
    loss_info = unreplicate(train_info["loss_info"])

    # total_transitions = 0
    # for i in range(config.num_updates):
    #     # summing total transitions per update from all devices
    #     total_transitions += (
    #         config.num_steps * config.num_envs_per_device * jax.local_device_count()
    #     )
    #     info = jtu.tree_map(lambda x: x[i].item(), loss_info)
    #     info["transitions"] = total_transitions
    #     wandb.log(info)

    run.summary["training_time"] = elapsed_time
    run.summary["steps_per_second"] = (
        config.total_timesteps_per_device * jax.local_device_count()
    ) / elapsed_time

    print("Final return: ", float(loss_info["eval/returns"][-1]))
    run.finish()


if __name__ == "__main__":
    train()
