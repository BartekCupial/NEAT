from typing import Tuple

import jax
import jax.numpy as jnp
from evojax.task.base import TaskState
from flax.struct import dataclass
from jax import random


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray


def sample_batch(key: jnp.ndarray, data: jnp.ndarray, labels: jnp.ndarray, batch_size: int) -> Tuple:
    # Ensure key has proper dimensionality
    # if key.ndim == 0:
    #     key = jnp.array([key, key])  # Convert scalar to proper key format
    # elif key.ndim == 1 and key.shape[0] != 2:
    #     key = jax.random.PRNGKey(int(key[0]))  # Regenerate proper key

    ix = random.choice(key=key, a=data.shape[0], shape=(batch_size,), replace=False)
    return (jnp.take(data, indices=ix, axis=0), jnp.take(labels, indices=ix, axis=0))


def loss_fn(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    per_example_losses = -jnp.sum(jax.nn.log_softmax(prediction, axis=-1) * jax.nn.one_hot(target, 2), axis=-1)
    return jnp.mean(per_example_losses)


def accuracy(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    predicted_class = jnp.argmax(prediction, axis=1)
    return jnp.mean(predicted_class == target)
