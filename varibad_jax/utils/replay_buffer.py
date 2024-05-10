"""
Based on https://github.com/ikostrikov/pynp-a2c-ppo-acktr

Used for on-policy rollout storages.
"""

from typing import Any, Tuple, Dict
import gym
import argparse
import numpy as np

import jax.numpy as jnp
import haiku as hk
from varibad_jax.agents.actor_critic import ActorCritic, ActorCriticInput


class OnlineStorage:
    def __init__(
        self,
        args: argparse.Namespace,
        num_steps: int,
        num_processes: int,
        state_dim: int,
        belief_dim: int,
        task_dim: int,
        action_space: gym.Space,
        hidden_size: int,
        latent_dim: int,
        normalise_rewards: bool,
    ):
        """
        Stores data collected during a rollout.

        Args:
            args (argparse.Namespace): Arguments.
            num_steps (int): Number of steps to store.
            num_processes (int): Number of parallel processes.
            state_dim (int): Dimensionality of state space.
            belief_dim (int): Dimensionality of belief space.
            task_dim (int): Dimensionality of task space.
            action_space (gym.Space): Action space.
            hidden_size (int): Size of hidden RNN states.
            latent_dim (int): Dimensionality of latent space (of VAE).
            normalise_rewards (bool): Whether to normalise rewards or not.
            use_popart (bool): Whether to use pop-art for reward normalisation.
            add_exploration_bonus (bool): Whether to add exploration bonus to
                the reward.
            intrinsic_reward (Any): Intrinsic reward object.
        """

        self.args = args
        # Support tuple state dimensions # TODO: update docstring accordingly
        if isinstance(state_dim, int):
            # We set int state_dim
            self.state_dim = (state_dim,)
        elif isinstance(state_dim, tuple) or isinstance(state_dim, list):
            # We set tuple state_dim
            self.state_dim = tuple(state_dim)
        else:
            raise ValueError(
                "state_dim must be int, tuple or list, got {}".format(type(state_dim))
            )
        self.belief_dim = belief_dim
        self.task_dim = task_dim

        self.num_steps = (
            num_steps  # how many steps to do per update (= size of online buffer)
        )
        self.num_processes = num_processes  # number of parallel processes
        self.step = 0  # keep track of current environment step

        # normalisation of the rewards
        self.normalise_rewards = normalise_rewards

        # inputs to the policy
        # this will include s_0 when state was reset (hence num_steps+1)
        self.prev_state = np.zeros((num_steps + 1, num_processes, *self.state_dim))
        if self.args.model.policy.pass_latent_to_policy:
            # latent variables (of VAE)
            self.latent_dim = latent_dim
            self.latent_samples = []
            self.latent_mean = []
            self.latent_logvar = []
            # hidden states of RNN (necessary if we want to re-compute embeddings)
            self.hidden_size = hidden_size
            self.hidden_states = np.zeros((num_steps + 1, num_processes, hidden_size))
        else:
            self.latent_mean = None
            self.latent_logvar = None
            self.latent_samples = None
        # next_state will include s_N when state was reset, skipping s_0
        # (only used if we need to re-compute embeddings after backpropagating RL loss through encoder)
        self.next_state = np.zeros((num_steps, num_processes, *self.state_dim))
        if self.args.model.policy.pass_belief_to_policy:
            self.beliefs = np.zeros((num_steps + 1, num_processes, belief_dim))
        else:
            self.beliefs = None

        self.tasks = np.zeros((num_steps + 1, num_processes, task_dim))

        # rewards and end of episodes
        self.rewards_raw = np.zeros((num_steps, num_processes, 1))
        self.rewards_normalised = np.zeros((num_steps, num_processes, 1))
        if self.args.model.use_hyperx_bonuses:
            self.hyperx_bonuses = np.zeros((num_steps, num_processes, 1))
            self.vae_recon_bonuses = np.zeros((num_steps, num_processes, 1))
        else:
            self.hyperx_bonuses = None
            self.vae_recon_bonuses = None

        self.done = np.zeros((num_steps + 1, num_processes, 1))
        self.masks = np.ones((num_steps + 1, num_processes, 1))
        # masks that indicate whether it's a true terminal state (false) or time limit end state (true)
        self.bad_masks = np.ones((num_steps + 1, num_processes, 1))

        # actions
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = np.zeros((num_steps, num_processes, action_shape))
        self.action_log_probs = None
        self.action_log_dist = None

        # values and returns
        self.value_preds = np.zeros((num_steps + 1, num_processes, 1))
        self.returns = np.zeros((num_steps + 1, num_processes, 1))
        self.denorm_value_preds = None

    @property
    def rewards(self):
        if self.normalise_rewards:
            return self.rewards_normalised
        else:
            return self.rewards_raw

    def insert(
        self,
        state: np.ndarray,
        belief: np.ndarray,
        task: np.ndarray,
        actions: np.ndarray,
        rewards_raw: np.ndarray,
        rewards_normalised: np.ndarray,
        hyperx_bonuses: np.ndarray,
        vae_recon_bonuses: np.ndarray,
        value_preds: np.ndarray,
        masks: np.ndarray,
        bad_masks: np.ndarray,
        done: np.ndarray,
        hidden_states: np.ndarray = None,
        latent_sample: np.ndarray = None,
        latent_mean: np.ndarray = None,
        latent_logvar: np.ndarray = None,
    ):
        """
        Insert a transition into the buffer.

        Args:
            state (np.ndarray)
            belief (np.ndarray)
            task (np.ndarray)
            actions (np.ndarray)
            rewards_raw (np.ndarray)
            rewards_normalised (np.ndarray)
            value_preds (np.ndarray)
            masks (np.ndarray): masks that indicate whether it's a true
                terminal state (false) or time limit end state (true)
            bad_masks (np.ndarray): masks that indicate whether it's a time
                limit end state because of failure
            done (np.ndarray):
            hidden_states (np.ndarray):
            latent_sample (np.ndarray)
            latent_mean (np.ndarray)
            latent_logvar (np.ndarray)
            level_seeds (np.ndarray)
        """
        self.prev_state[self.step + 1] = state
        if self.args.model.policy.pass_belief_to_policy:
            self.beliefs[self.step + 1] = belief
        if self.args.model.policy.pass_task_to_policy:
            self.tasks[self.step + 1] = task
        if self.args.model.policy.pass_latent_to_policy:
            self.latent_samples.append(latent_sample.copy())
            self.latent_mean.append(latent_mean.copy())
            self.latent_logvar.append(latent_logvar.copy())
            # self.hidden_states[self.step + 1] = hidden_states
        self.actions[self.step] = actions.copy()
        self.rewards_raw[self.step] = rewards_raw
        self.rewards_normalised[self.step] = rewards_normalised
        if self.args.model.use_hyperx_bonuses:
            self.hyperx_bonuses[self.step] = hyperx_bonuses
            self.vae_recon_bonuses[self.step] = vae_recon_bonuses

        if isinstance(value_preds, list):
            self.value_preds[self.step] = value_preds[0]
        else:
            self.value_preds[self.step] = value_preds
        self.masks[self.step + 1] = masks
        self.bad_masks[self.step + 1] = bad_masks
        self.done[self.step + 1] = done
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        """
        Copy the last state of the buffer to the first state.
        """
        self.prev_state[0] = self.prev_state[-1]
        if self.args.model.policy.pass_belief_to_policy:
            self.beliefs[0] = self.beliefs[-1]
        if self.args.model.policy.pass_task_to_policy:
            self.tasks[0] = self.tasks[-1]
        if self.args.model.policy.pass_latent_to_policy:
            self.latent_samples = []
            self.latent_mean = []
            self.latent_logvar = []
            # self.hidden_states[0] = self.hidden_states[-1]

        self.done[0] = self.done[-1]
        self.masks[0] = self.masks[-1]
        self.bad_masks[0] = self.bad_masks[-1]
        self.action_log_probs = None
        self.action_log_dist = None

    def compute_returns(
        self,
        next_value: np.ndarray,
        use_gae: bool,
        gamma: float,
        tau: float,
        use_proper_time_limits=True,
    ):
        """
        Compute the returns for each step in the buffer.

        Args:
            next_value: (np.ndarray): The value of the next state.
            use_gae: (bool): Whether to use Generalised Advantage Estimation.
            gamma: (float): The discount factor.
            tau: (float): The GAE parameter.
            use_proper_time_limits: (bool): Whether to use proper time limits.
            vae: (VAE): The VAE used for latent space.
        """
        rewards = (
            self.rewards_normalised.copy()
            if self.normalise_rewards
            else self.rewards_raw.copy()
        )

        if self.args.model.use_hyperx_bonuses:
            rewards += self.hyperx_bonuses
            rewards += self.vae_recon_bonuses

        self._compute_returns(
            next_value=next_value,
            rewards=rewards,
            value_preds=self.value_preds,
            returns=self.returns,
            gamma=gamma,
            tau=tau,
            use_gae=use_gae,
            use_proper_time_limits=use_proper_time_limits,
        )

    def _compute_returns(
        self,
        next_value,
        rewards,
        value_preds,
        returns,
        gamma,
        tau,
        use_gae,
        use_proper_time_limits,
    ):
        if use_proper_time_limits:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.shape[0])):
                    delta = (
                        rewards[step]
                        + gamma * value_preds[step + 1] * self.masks[step + 1]
                        - value_preds[step]
                    )
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.shape[0])):
                    returns[step] = (
                        returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]
                    ) * self.bad_masks[step + 1] + (
                        1 - self.bad_masks[step + 1]
                    ) * value_preds[
                        step
                    ]
        else:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.shape[0])):
                    delta = (
                        rewards[step]
                        + gamma * value_preds[step + 1] * self.masks[step + 1]
                        - value_preds[step]
                    )
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.shape[0])):
                    returns[step] = (
                        returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]
                    )

    def num_transitions(self):
        """
        Get the total number of transitions in the buffer.

        Returns:
            (int) The total number of transitions in the buffer.
        """
        return len(self.prev_state) * self.num_processes

    def before_update(self, agent, rng_key):
        if self.latent_mean is not None:
            latent = np.concatenate(
                [self.latent_mean[:-1], self.latent_logvar[:-1]], axis=-1
            )
        else:
            latent = None

        if self.task_dim > 0:
            task = self.tasks[:-1]
        else:
            task = None

        if self.args.model.policy.use_rnn_policy:
            policy_output, policy_state = agent.jit_unroll(
                agent._params,
                agent._state,
                rng_key,
                agent,
                env_state=self.prev_state[:-1],
                latent=latent,
                task=task,
            )
        else:
            (policy_output, _), policy_state = agent.get_action(
                rng_key,
                env_state=self.prev_state[:-1],
                latent=latent,
                task=task,
                is_training=True,
            )

        log_probs = policy_output.dist.log_prob(self.actions.astype(np.int32).squeeze())

        # import jax

        # jax.debug.breakpoint()
        self.action_log_probs = log_probs
        # [T, B]

        if len(self.action_log_probs.shape) == 2:
            self.action_log_probs = np.expand_dims(self.action_log_probs, axis=-1)

        return policy_state
