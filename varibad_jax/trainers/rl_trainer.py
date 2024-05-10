import time
from absl import logging
import jax
import jax.numpy as jnp
import tqdm
from ml_collections.config_dict import ConfigDict
from flax import struct
import jax.tree_util as jtu
from typing import Optional

from varibad_jax.trainers.base_trainer import BaseTrainer
from varibad_jax.agents.ppo.ppo import PPOAgent
from varibad_jax.utils.rollout import run_rollouts
import varibad_jax.utils.general_utils as gutl


class Transition(struct.PyTreeNode):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    # for rnn policy
    prev_action: jax.Array
    prev_reward: jax.Array
    task: Optional[jax.Array] = None


def calculate_gae(
    transitions: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:

    # single iteration for the loop
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = (
            transition.reward
            + gamma * next_value * (1 - transition.done)
            - transition.value
        )
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value


class RLTrainer(BaseTrainer):
    def __init__(self, config: ConfigDict):
        super().__init__(config)
        self.total_steps = 0
        self.num_updates = config.num_frames // config.num_steps // config.num_envs

        logging.info(f"num rl updates = {self.num_updates}")
        logging.info(f"steps per rollout = {self.steps_per_rollout}")
        logging.info(f"action_dim = {self.action_dim}")

        self.agent = PPOAgent(
            config=config.model,
            observation_shape=self.envs.observation_space.shape,
            action_dim=self.action_dim,
            input_action_dim=self.input_action_dim,
            continuous_actions=self.continuous_actions,
            key=next(self.rng_seq),
        )

        def _env_step(runner_state, _):
            rng, ts, prev_xtimestep, prev_action, prev_reward, prev_hstate = (
                runner_state
            )

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            observation = prev_xtimestep.timestep.observation
            observation = observation.astype(jnp.float32)

            # GET CURRENT TASK VECTOR
            if config.env.env_name == "gridworld":
                task = prev_xtimestep.timestep.state.goal
            elif config.env.env_name == "xland":
                task = prev_xtimestep.timestep.state.goal_encoding

            if len(task.shape) == 1:
                task = task[jnp.newaxis]
            task = task.astype(jnp.float32)

            (policy_output, hstate), new_state = self.agent.get_action_jit(
                ts,
                _rng,
                env_state=observation,
                hidden_state=prev_hstate,
                task=task,
                is_training=True,
            )

            action = policy_output.action
            log_prob = policy_output.log_prob
            value = policy_output.value

            # STEP ENV
            xtimestep = jax.vmap(self.envs.step, in_axes=(None, 0, 0))(
                self.env_params, prev_xtimestep, action
            )
            reward = xtimestep.timestep.reward
            action = action.reshape(-1, 1).astype(jnp.float32)
            reward = reward.reshape(-1, 1)
            done = xtimestep.timestep.last().reshape(-1, 1)

            # reset the hidden state when the episode is over
            mask = xtimestep.timestep.last()[:, None]
            if hstate is not None:
                hstate = hstate * (1 - mask)

            transition = Transition(
                done=done,
                action=action,
                value=value,
                reward=reward,
                log_prob=log_prob,
                obs=prev_xtimestep.timestep.observation.astype(jnp.float32),
                prev_action=action,
                prev_reward=prev_reward,
                task=task,
            )

            ts.state = new_state
            runner_state = (rng, ts, xtimestep, action, reward, hstate)
            return runner_state, transition

        self._env_step = _env_step

        def _update_epoch(update_state, _):
            (
                rng,
                ts,
                init_hstate,
                transitions,
                advantages,
                targets,
            ) = update_state

            # MINIBATCHES PREPARATION
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, config.num_envs)

            # [T, B, ...]
            batch = (init_hstate, transitions, advantages, targets)
            # [B, T, ...], as our model assumes
            batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

            shuffled_batch = jtu.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            # [num_minibatches, minibatch_size, ...]
            minibatches = jtu.tree_map(
                lambda x: jnp.reshape(
                    x, (config.model.num_minibatches, -1) + x.shape[1:]
                ),
                shuffled_batch,
            )

            # import ipdb

            # ipdb.set_trace()

            rng, _rng = jax.random.split(rng)
            minibatch_carry = (_rng, ts)
            # PERFORM GRADIENT UPDATE FOR EACH MINIBATCH
            minibatch_carry, update_info = jax.lax.scan(
                self.agent._update_minibatch, minibatch_carry, minibatches
            )

            (rng, ts) = minibatch_carry

            update_state = (
                rng,
                ts,
                init_hstate,
                transitions,
                advantages,
                targets,
            )
            return update_state, update_info

        self._update_epoch = _update_epoch

    def train(self):
        logging.info("Training starts")

        # FIRST EVAL
        if not self.config.skip_first_eval:
            eval_metrics, *_ = run_rollouts(
                rng=next(self.rng_seq),
                agent=self.agent,
                env=self.eval_envs,
                config=self.config,
                steps_per_rollout=self.steps_per_rollout,
                action_dim=self.input_action_dim,
                wandb_run=self.wandb_run,
            )

        # INIT ENV
        prev_action = jnp.zeros(
            (self.config.num_envs, self.input_action_dim), dtype=jnp.float32
        )
        prev_reward = jnp.zeros((self.config.num_envs, 1))
        if self.config.model.policy.use_rnn_policy:
            init_hstate = jnp.zeros(
                (self.config.num_envs, self.config.model.policy.rnn_hidden_size)
            )
        else:
            init_hstate = None

        reset_rngs = jax.random.split(next(self.rng_seq), self.config.num_envs)
        xtimestep = self.jit_reset(self.env_params, reset_rngs)

        for self.iter_idx in tqdm.tqdm(
            range(self.num_updates),
            smoothing=0.1,
            desc="rl training",
            disable=self.config.disable_tqdm,
        ):
            runner_state = (
                next(self.rng_seq),
                self.agent._ts,
                xtimestep,
                prev_action,
                prev_reward,
                init_hstate,
            )

            initial_hstate = runner_state[-1]

            steps_start = time.time()
            # transitions: [T, B ...]
            # COLLECT TRANSITIONS
            runner_state, transitions = jax.lax.scan(
                self._env_step, runner_state, None, self.config.num_steps
            )

            steps_end = time.time()
            fps = (self.config.num_envs * self.config.num_steps) / (
                steps_end - steps_start
            )
            # logging.info(f"fps = {fps}")

            # CALCULATE ADVANTAGE
            rng, ts, xtimestep, prev_action, prev_reward, hstate = runner_state
            self.agent._ts = ts

            # calculate value of the last step for bootstrapping
            (policy_output, _), state = self.agent.get_action(
                next(self.rng_seq),
                env_state=xtimestep.timestep.observation.astype(jnp.float32),
                hidden_state=hstate,
                task=transitions.task[-1],  # get the most recent task
                is_training=True,
            )

            # update training state (state)
            self.agent._ts.state = state
            last_val = policy_output.value

            advantages, targets = calculate_gae(
                transitions,
                last_val,
                self.config.model.gamma,
                self.config.model.tau,
            )

            if initial_hstate is not None:
                init_hstate = initial_hstate[None, :]

            update_state = (
                next(self.rng_seq),
                self.agent._ts,
                init_hstate,
                transitions,
                advantages,
                targets,
            )

            # UPDATE POLICY
            update_state, loss_info = jax.lax.scan(
                self._update_epoch, update_state, None, self.config.model.num_epochs
            )
            # UPDATE AGENT STATES
            self.agent._ts = update_state[1]

            # averaging over minibatches then over epochs
            loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

            init_hstate = hstate  # SET HIDDEN STATE FOR NEXT ROLLOUTS

            # LOGGING
            if ((self.iter_idx + 1) % self.config.log_interval) == 0:
                # logging.info(f"ep_rew: {transitions.reward.sum(0).mean()}")
                metrics = {**loss_info, "time/fps": fps}
                if self.wandb_run is not None:
                    self.wandb_run.log(metrics, step=self.iter_idx)

            # EVALUATION
            if ((self.iter_idx + 1) % self.config.eval_interval) == 0:
                eval_metrics, *_ = run_rollouts(
                    rng=next(self.rng_seq),
                    agent=self.agent,
                    env=self.eval_envs,
                    config=self.config,
                    steps_per_rollout=self.steps_per_rollout,
                    action_dim=self.input_action_dim,
                    wandb_run=self.wandb_run,
                )
                # SAVE MODEL
                self.save_model(self.agent.save_dict, eval_metrics, iter=self.iter_idx)

                print(eval_metrics)
                if self.wandb_run is not None:
                    eval_metrics = gutl.prefix_dict_keys(eval_metrics, "eval/")
                    self.wandb_run.log(eval_metrics)
