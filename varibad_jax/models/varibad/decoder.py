import dataclasses
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
from varibad_jax.models.common import ImageEncoder


@dataclasses.dataclass
class RewardDecoder(hk.Module):
    """RewardDecoder"""

    def __init__(
        self,
        image_obs: bool = False,
        input_action: bool = False,
        input_prev_state: bool = False,
        embedding_dim: int = 8,
        layer_sizes: List[int] = [32, 32],
        image_encoder_config: dict = None,
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs
    ):
        """Decodes the rewards in a trajectory given the latent information

        Args:
          output_size: number of output units of the final MLP
          input_action: condition the decoding on action
          input_prev_state: condition the decoding on the prev state
          embedding_dim: hidden size for embedding inputs
          layer_sizes: a list of integers specifying the size of each layer in the
            output MLP
        """
        super().__init__(name="Decoder")

        init_kwargs = dict(w_init=w_init, b_init=b_init)

        self.image_obs = image_obs
        self.input_action = input_action
        self.input_prev_state = input_prev_state
        self.embedding_dim = embedding_dim

        self.latent_embed = hk.Linear(
            self.embedding_dim, name="latent_embed", **init_kwargs
        )

        if self.image_obs:
            self.state_embed = ImageEncoder(**image_encoder_config, **init_kwargs)
        else:
            self.state_embed = hk.Linear(
                self.embedding_dim, name="state_embed", **init_kwargs
            )

        if self.input_action:
            self.action_embed = hk.Linear(
                self.embedding_dim, name="action_embed", **init_kwargs
            )

        self.output_mlp = hk.nets.MLP(
            list(layer_sizes) + [1],
            activation=nn.gelu,
            name="reward_mlp",
            activate_final=False,
            **init_kwargs,
        )

    def __call__(
        self,
        latents: jnp.ndarray,
        prev_states: jnp.ndarray = None,
        next_states: jnp.ndarray = None,
        actions: jnp.ndarray = None,
        is_training: bool = True,
    ):
        """Forwards the reward decoder network"""
        latent_embed = self.latent_embed(latents)
        latent_embed = nn.gelu(latent_embed)

        if self.image_obs:
            next_state_embed = self.state_embed(next_states, is_training=is_training)
        else:
            next_state_embed = self.state_embed(next_states)

        next_state_embed = nn.gelu(next_state_embed)
        inputs = jnp.concatenate((latent_embed, next_state_embed), axis=-1)

        if self.input_prev_state:
            prev_state_embed = self.state_embed(prev_states)
            prev_state_embed = nn.gelu(prev_state_embed)
            inputs = jnp.concatenate((inputs, prev_state_embed), axis=-1)

        if self.input_action:
            action = self.action_embed(actions)
            action = nn.gelu(action)
            inputs = jnp.concatenate((inputs, action), axis=-1)

        rew_pred = self.output_mlp(inputs)
        return rew_pred

@dataclasses.dataclass
class TaskDecoder(hk.Module):
    def __init__(
        self,
        task_dim: int = 2,
        layer_sizes: List[int] = [32, 32],
        w_init=hk.initializers.VarianceScaling(scale=2.0),
        b_init=hk.initializers.Constant(0.0),
        **kwargs
    ):
        """Decodes the tasks in a trajectory given the latent information

        Args:
          embedding_dim: hidden size for embedding inputs
          layer_sizes: a list of integers specifying the size of each layer in the
            output MLP
        """
        super().__init__(name="TaskDecoder")

        init_kwargs = dict(w_init=w_init, b_init=b_init)
        self.output_mlp = hk.nets.MLP(
            list(layer_sizes) + [task_dim],
            activation=nn.gelu,
            name="task_mlp",
            activate_final=False,
            **init_kwargs,
        )
    
    def __call__(self, latents: jnp.ndarray, is_training: bool = True):
        """Forwards the task decoder network"""
        task_pred = self.output_mlp(latents)
        return task_pred