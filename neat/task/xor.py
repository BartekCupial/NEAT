import sys
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from evojax.task.base import TaskState, VectorizedTask
from flax.struct import dataclass
from jax import random


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray


def mse_loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    return jnp.mean((prediction - target) ** 2)


def accuracy(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    predicted_class = jnp.argmax(prediction, axis=1)
    return jnp.mean(predicted_class == target)


class XOR(VectorizedTask):
    """XOR task."""

    def __init__(
        self,
        batch_size: int = 1024,
        test: bool = False,
    ):
        self.max_steps = 1
        self.obs_shape = tuple(
            2,
        )
        self.act_shape = tuple(
            1,
        )

        key = random.PRNGKey(0)
        train_key, test_key = random.split(key)

        # Training data (fixed)
        X_train = random.uniform(train_key, (batch_size, 2), minval=-1, maxval=1)
        self.train_labels = (X_train[:, 0] * X_train[:, 1] > 0).astype(int)
        self.train_data = X_train

        # Test data (different fixed set)
        X_test = random.uniform(test_key, (batch_size, 2), minval=-1, maxval=1)
        self.test_labels = (X_test[:, 0] * X_test[:, 1] > 0).astype(int)
        self.test_data = X_test

        def reset_fn(key):
            """Return fixed dataset, ignoring input key."""
            if test:
                return State(obs=self.test_data, labels=self.test_labels)
            return State(obs=self.train_data, labels=self.train_labels)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            if test:
                reward = accuracy(action, state.labels)
            else:
                reward = -mse_loss(action, state.labels)
            return state, reward, jnp.ones(())

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self, state: TaskState, action: jnp.ndarray) -> Tuple[TaskState, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)
