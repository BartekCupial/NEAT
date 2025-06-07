from typing import Tuple

import jax
import jax.numpy as jnp
from evojax.task.base import TaskState, VectorizedTask
from jax import random

from neat.task.util import State, accuracy, loss_fn, sample_batch


def generate_data(key, dataset_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    X = random.uniform(key, (dataset_size, 5))

    # Generate random angles
    t = jnp.sqrt(X[:, 0]) * 4 * jnp.pi

    # Spiral 1 (class 0)
    r_a = 2 * t
    x1_a = r_a * jnp.cos(t) + X[:, 1] * 5.0
    x2_a = r_a * jnp.sin(t) + X[:, 2] * 5.0

    # Spiral 2 (class 1), offset by pi
    r_b = 2 * t + 2 * jnp.pi
    x1_b = r_b * jnp.cos(t) + X[:, 3] * 5.0
    x2_b = r_b * jnp.sin(t) + X[:, 4] * 5.0

    X_train = jnp.concatenate([jnp.stack([x1_a, x2_a], axis=1), jnp.stack([x1_b, x2_b], axis=1)], axis=0)
    y_train = jnp.concatenate([jnp.zeros(X_train.shape[0] // 2), jnp.ones(X_train.shape[0] // 2)], axis=0)
    return X_train, y_train


class Spiral(VectorizedTask):
    """Spiral task."""

    def __init__(
        self,
        batch_size: int = 10,
        dataset_size: int = 1000,
        test: bool = False,
    ):
        self.max_steps = 1
        self.obs_shape = tuple((2,))
        self.act_shape = tuple((2,))

        key = random.PRNGKey(0)
        train_key, test_key = random.split(key)

        # Training data (fixed)
        self.train_data, self.train_labels = generate_data(train_key, dataset_size)

        # Test data (different fixed set)
        self.test_data, self.test_labels = generate_data(test_key, dataset_size)

        def reset_fn(key):
            """Return fixed dataset, ignoring input key."""
            if test:
                batch_data, batch_labels = self.test_data, self.test_labels
            else:
                batch_data, batch_labels = sample_batch(key, self.train_data, self.train_labels, batch_size)
            return State(obs=batch_data, labels=batch_labels)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))
        # self._reset_fn = jax.jit(reset_fn)

        def step_fn(state, action):
            if test:
                reward = accuracy(action, state.labels)
            else:
                reward = -loss_fn(action, state.labels)
            return state, reward, jnp.ones(())

        self._step_fn = jax.jit(jax.vmap(step_fn))
        # self._step_fn = jax.jit(step_fn)

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self, state: TaskState, action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
