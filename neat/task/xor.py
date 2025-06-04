from typing import Tuple

import jax
import jax.numpy as jnp
from evojax.task.base import TaskState, VectorizedTask
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


class XOR(VectorizedTask):
    """XOR task."""

    def __init__(
        self,
        batch_size: int = 10,
        dataset_size: int = 1000,
        test: bool = False,
    ):
        self.max_steps = 1
        self.obs_shape = tuple((2,))
        self.act_shape = tuple((1,))

        key = random.PRNGKey(0)
        train_key, test_key = random.split(key)

        # Training data (fixed)
        X_train = random.uniform(train_key, (dataset_size, 2), minval=-1, maxval=1)
        self.train_labels = (X_train[:, 0] * X_train[:, 1] > 0).astype(int)
        self.train_data = X_train

        # Test data (different fixed set)
        X_test = random.uniform(test_key, (dataset_size, 2), minval=-1, maxval=1)
        self.test_labels = (X_test[:, 0] * X_test[:, 1] > 0).astype(int)
        self.test_data = X_test

        def reset_fn(key):
            """Return fixed dataset, ignoring input key."""
            if test:
                batch_data, batch_labels = self.test_data, self.test_labels
            else:
                batch_data, batch_labels = sample_batch(key, self.train_data, self.train_labels, batch_size)
            return State(obs=batch_data, labels=batch_labels)

        # self._reset_fn = jax.jit(jax.vmap(reset_fn))
        self._reset_fn = jax.jit(reset_fn)

        def step_fn(state, action):
            if test:
                reward = accuracy(action, state.labels)
            else:
                reward = -loss_fn(action, state.labels)
            return state, reward, jnp.ones(())

        # self._step_fn = jax.jit(jax.vmap(step_fn))
        self._step_fn = jax.jit(step_fn)

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self, state: TaskState, action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
