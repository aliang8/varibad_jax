from typing import Tuple
import jax
import numpy as np
import haiku as hk
import optax
import jax.numpy as jnp
from absl import logging
from functools import partial

from varibad_jax.models.base import BaseAgent
from varibad_jax.models.lapo.model import LatentFDM, LatentActionIDM


class LAPOAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def net(self, states, actions, is_training=False):
        idm = LatentActionIDM(self.config.policy.idm)
        fdm = LatentFDM(self.config.policy.fdm)

        idm_output = idm(states, actions, is_training=is_training)
        fdm_output = fdm(states, actions, is_training=is_training)

    def _init_model(self):
        t, bs = 2, 2
        dummy_states = np.zeros((t, bs, *self.observation_shape), dtype=np.float32)
        dummy_actions = np.zeros((t, bs, self.input_action_dim))

        self._params, self._state = self.model.init(
            self._key,
            states=dummy_states,
            actions=dummy_actions,
            is_training=True,
        )

    def loss_fn(self):
        pass
