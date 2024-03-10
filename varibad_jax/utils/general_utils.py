import collections
from typing import Any, Optional
from absl import logging
from flax.training import orbax_utils
import haiku as hk
import jax
import jax.numpy as jnp
import orbax
import numpy as np
import tensorflow_probability
import tensorflow as tf

tfp = tensorflow_probability.substrates.jax
tfd = tensorflow_probability.substrates.jax.distributions


def format_dict_keys(dictionary, format_fn):
    """Returns new dict with `format_fn` applied to keys in `dictionary`."""
    return collections.OrderedDict(
        [(format_fn(key), value) for key, value in dictionary.items()]
    )


def prefix_dict_keys(dictionary, prefix):
    """Add `prefix` to keys in `dictionary`."""
    return format_dict_keys(dictionary, lambda key: "%s%s" % (prefix, key))


def compute_all_grad_norm(grad_norm_type, grads):
    grad_norms = {}

    for mod, name, v in hk.data_structures.traverse(grads):
        grad_norms[mod + "/" + name] = compute_grad_norm(grad_norm_type, v)
        if jnp.isnan(v):
            jax.debug.breakpoint()

    stats = {}
    max_grad_norm = jnp.max(jnp.array(list(grad_norms.values())))
    average_grad_norm = jnp.mean(jnp.array(list(grad_norms.values())))
    stats["max_grad_norm"] = max_grad_norm
    stats["average_grad_norm"] = average_grad_norm
    return grad_norms, stats


def compute_grad_norm(grad_norm_type, grads):
    """Compute norm of the gradients.

    Args:
      grad_norm_type: type of norm to calculate {1,2,inf}
      grads: gradient

    Returns:
      gradient norms
    """
    if grad_norm_type is not None:
        grad_norm_type_to_ord = {"1": 1, "2": 2, "inf": jnp.inf}
        grad_type = grad_norm_type_to_ord[grad_norm_type]
        grad_norms = jnp.linalg.norm(grads, ord=grad_type)
    else:
        # It will be easier to manage downstream if we just fill this with zeros.
        # Rather than have this be potentially a None type.
        grad_norms = jnp.zeros_like(grads[:, 0])
    return grad_norms


class AttrDict(dict):
    """Allows to access dict keys as obj.foo in addition

    to the traditional way obj['foo']"

    Example:
        >>> d = AttrDict(foo=1, bar=2)
        >>> assert d["foo"] == d.foo
        >>> d.bar = "hello"
        >>> assert d.bar == "hello"
    """

    def __init__(self, *args, **kwargs):
        """:param args: Passthrough arguments for standard dict.

        :param kwargs: Passthrough keyword arguments for standard dict.
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        """:return: Copy of AttrDict"""
        newd = super().copy()
        return self.__class__(**newd)


class CheckpointSaver:

    def __init__(self, ckpt_dir: str, max_to_keep: Optional[int] = None):
        # setup saving
        # for saving checkpoints and reloading from checkpoint
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        # save every two
        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=max_to_keep, save_interval_steps=1
        )
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            ckpt_dir, orbax_checkpointer, options
        )

    def get_latest_ckpt(self):
        return self.checkpoint_manager.latest_step()

    def load_from_ckpt(self, iter_idx: int = None):
        if iter_idx is None:
            step = self.get_latest_ckpt()
        else:
            step = iter_idx

        print(f"loading from ckpt: {step}")
        logging.info(f"loading model from ckpt, step = {step}")
        ckpt = self.checkpoint_manager.restore(step)
        return ckpt

    def save(self, iter_idx: int, ckpt: dict[str, Any]):
        # Bundle everything together.
        print("Saving model checkpoint ...")
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpoint_manager.save(
            iter_idx, ckpt, save_kwargs={"save_args": save_args}
        )
