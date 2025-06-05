from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
import pytest

from neat.algo.genome import NEATGenome
from neat.policy import NEATPolicy
from neat.task import XOR


def loss_fn(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    per_example_losses = -jnp.sum(jax.nn.log_softmax(prediction, axis=-1) * jax.nn.one_hot(target, 2), axis=-1)
    return jnp.mean(per_example_losses)


def accuracy(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    predicted_class = jnp.argmax(prediction, axis=1)
    return jnp.mean(predicted_class == target)


class TestXOR(object):
    def test_xor(self, fully_connected_genome: NEATGenome):
        config = SimpleNamespace(
            **{
                "batch_size": 32,
                "dataset_size": 1024,
                "max_iter": 1000,
                "log_interval": 100,
                "learning_rate": 0.1,
                "seed": 42,
                "gpu_id": 0,
                "debug": False,
            }
        )

        # Initialize model and optimizer
        model = NEATPolicy()
        params, static_params = model.compile_genome(fully_connected_genome)

        key = jax.random.PRNGKey(0)
        tx = optax.adam(config.learning_rate)
        opt_state = tx.init(params)

        train_task = XOR(batch_size=config.batch_size, dataset_size=config.dataset_size, test=False)
        test_task = XOR(batch_size=config.batch_size, dataset_size=config.dataset_size, test=True)

        def model_loss_for_grad(model_params, obs_batch, labels_batch):
            predictions = model.apply(model_params, static_params, obs_batch)
            return loss_fn(predictions, labels_batch)

        @jax.jit
        def update(current_params, current_opt_state, obs_batch, labels_batch):
            loss_value, grads = jax.value_and_grad(model_loss_for_grad)(current_params, obs_batch, labels_batch)
            updates, new_opt_state = tx.update(grads, current_opt_state)
            new_params = optax.apply_updates(current_params, updates)
            return new_params, new_opt_state, loss_value

        # Training loop
        for epoch in range(config.max_iter):
            key, subkey_train = jax.random.split(key)
            # Assuming train_task.reset provides state.obs and state.labels
            state = train_task.reset(subkey_train)

            params, opt_state, loss = update(params, opt_state, state.obs, state.labels)

            if epoch % config.log_interval == 0:  # Use log_interval from config
                # Evaluate on test set
                key, subkey_test = jax.random.split(key)
                test_state = test_task.reset(subkey_test)

                # Get raw predictions (logits) from the model
                test_logits = model.apply(params, static_params, test_state.obs)
                # Calculate accuracy using logits
                acc = accuracy(test_logits, test_state.labels)

        assert acc > 0.9, f"Accuracy {acc} is below threshold for XOR task"
