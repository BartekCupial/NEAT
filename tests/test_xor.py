from types import SimpleNamespace

import jax
import optax

from neat.algo.genome import NEATGenome
from neat.policy import NEATPolicy
from neat.task import XOR


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
        policy = NEATPolicy()
        params = policy.compile_population([fully_connected_genome])
        diff_params, static_params = params

        key = jax.random.PRNGKey(0)
        tx = optax.adam(config.learning_rate)
        opt_state = tx.init(diff_params)

        train_task = XOR(batch_size=config.batch_size, dataset_size=config.dataset_size, test=False)
        test_task = XOR(batch_size=config.batch_size, dataset_size=config.dataset_size, test=True)

        def model_loss_for_grad(model_params, task_state):
            full_params = (model_params, static_params)
            actions, _ = policy.get_actions(task_state, full_params, None)
            task_state, scores, done = train_task.step(task_state, actions)

            return -scores.mean()

        @jax.jit
        def update(current_params, current_opt_state, task_state):
            loss_value, grads = jax.value_and_grad(model_loss_for_grad)(current_params, task_state)
            updates, new_opt_state = tx.update(grads, current_opt_state)
            new_params = optax.apply_updates(current_params, updates)
            return new_params, new_opt_state, loss_value

        # Training loop
        for epoch in range(config.max_iter):
            key, subkey = jax.random.split(key)
            reset_keys = jax.random.split(subkey, 1)
            # Assuming train_task.reset provides state.obs and state.labels
            train_state = train_task.reset(reset_keys)

            diff_params, opt_state, loss = update(diff_params, opt_state, train_state)

            if epoch % config.log_interval == 0:  # Use log_interval from config
                # Evaluate on test set
                key, subkey = jax.random.split(key)
                reset_keys = jax.random.split(subkey, 1)
                test_state = test_task.reset(reset_keys)

                full_params = (diff_params, static_params)
                actions, _ = policy.get_actions(test_state, full_params, None)
                test_state, accuracy, done = test_task.step(test_state, actions)

        assert accuracy > 0.9, f"Accuracy {accuracy} is below threshold for XOR task"
