from typing import Tuple

import jax
import jax.numpy as jnp
from evojax.task.base import TaskState, VectorizedTask
from jax import random

from neat.task.util import State, accuracy, loss_fn, sample_batch


class Circle(VectorizedTask):
    """Circle task."""

    def __init__(
        self,
        batch_size: int = 10,
        dataset_size: int = 1000,
        test: bool = False,
        radius=0.7,
    ):
        self.max_steps = 1
        self.obs_shape = tuple((2,))
        self.act_shape = tuple((2,))

        key = random.PRNGKey(0)
        train_key, test_key = random.split(key)

        # Training data (fixed)
        X_train = random.uniform(train_key, (dataset_size, 2), minval=-1, maxval=1)
        self.train_labels = (X_train[:, 0] ** 2 * X_train[:, 1] ** 2 < radius**2).astype(int)
        self.train_data = X_train

        # Test data (different fixed set)
        X_test = random.uniform(test_key, (dataset_size, 2), minval=-1, maxval=1)
        self.test_labels = (X_test[:, 0] ** 2 * X_test[:, 1] ** 2 < radius**2).astype(int)
        self.test_data = X_test

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
