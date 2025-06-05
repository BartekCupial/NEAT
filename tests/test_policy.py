from typing import Dict

import jax
import jax.numpy as jnp
import optax
import pytest

from neat.algo.genome import NEATGenome
from neat.policy import NEATPolicy


class TestPolicy(object):
    def test_compile_policy(self, genome1: NEATGenome):
        """Test the policy compilation."""
        policy = NEATPolicy()
        policy.compile_genome(genome1)

    def test_forward_pass(self, genome1: NEATGenome):
        """Test the forward pass through the policy."""
        policy = NEATPolicy()
        params, static_params = policy.compile_genome(genome1)
        inputs = jnp.array([[0.5, 0.5, 0.5]])

        outputs = policy.apply(params, static_params, inputs)

        assert jnp.all(outputs == 0.75)

    def test_gradients(self, genome1: NEATGenome):
        """Test the gradients of the policy."""
        policy = NEATPolicy()
        params, static_params = policy.compile_genome(genome1)

        inputs = jnp.array([[0.5, 0.5, 0.5]])
        targets = jnp.array([[1.0]])

        def loss_fn(predictions, targets):
            return jnp.mean((predictions - targets) ** 2)

        def model_loss_for_grad(model_params, obs_batch, labels_batch):
            predictions = policy.apply(model_params, static_params, obs_batch)
            return loss_fn(predictions, labels_batch)

        grads = jax.grad(model_loss_for_grad)(params, inputs, targets)

        # Gradient Calculation
        # With prediction = 0.75, target = 1.0:
        #     Loss: (0.75 - 1.0)² = 0.0625
        #     ∂Loss/∂output: 2 × (0.75 - 1.0) = -0.5
        # Gradients for connections to output node 4:
        #     ∂Loss/∂w(1→4): -0.5 × input = -0.5 × 0.5 = -0.25
        #     ∂Loss/∂w(3→4): -0.5 × input = -0.5 × 0.5 = -0.25
        #     ∂Loss/∂w(5→4): -0.5 × hidden = -0.5 × 0.5 = -0.25
        # Gradients for connections to hidden node 5:
        #     ∂Loss/∂w(2→5): -0.25 × input = -0.25 × 0.5 = -0.125
        #     ∂Loss/∂w(1→5): -0.25 × input = -0.25 × 0.5 = -0.125

        assert grads is not None
        assert grads["weights"][4, 0] == -0.25  # w(1→4)
        assert grads["weights"][4, 2] == -0.25  # w(3→4)
        assert grads["weights"][4, 3] == -0.25  # w(5→4)
        assert grads["weights"][3, 0] == -0.125  # w(1→5)
        assert grads["weights"][3, 1] == -0.125  # w(2→5)
        assert grads["weights"][4, 1] == 0.0  # w(2→4) should not change since it's disabled

    def test_train(self, genome1: NEATGenome):
        """Test the training process of the policy."""
        policy = NEATPolicy()
        params, static_params = policy.compile_genome(genome1)
        inputs = jnp.array([[0.5, 0.5, 0.5]])
        targets = jnp.array([[1.0]])

        tx = optax.adam(learning_rate=0.01)
        opt_state = tx.init(params)

        def loss_fn(predictions, targets):
            return jnp.mean((predictions - targets) ** 2)

        def model_loss_for_grad(model_params, obs_batch, labels_batch):
            predictions = policy.apply(model_params, static_params, obs_batch)
            return loss_fn(predictions, labels_batch)

        @jax.jit
        def update(current_params, current_opt_state, obs_batch, labels_batch):
            loss_value, grads = jax.value_and_grad(model_loss_for_grad)(current_params, obs_batch, labels_batch)
            updates, new_opt_state = tx.update(grads, current_opt_state)
            new_params = optax.apply_updates(current_params, updates)
            return new_params, new_opt_state, loss_value

        for i in range(100):
            params, opt_state, loss = update(params, opt_state, inputs, targets)

        assert loss < 0.01

        mask = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 1, 1, 0],
            ]
        )

        assert (
            jnp.sum(params["weights"] * jnp.logical_not(mask)) == 0
        ), "Weights for disabled connections should be zero"
        assert jnp.array_equal(params["weights"] != 0, mask), "Weights for enabled connections should not be zero"

    def test_population(self, genome1: NEATGenome, genome2: NEATGenome):
        """Test the compilation of a population of genomes."""
        policy = NEATPolicy()
        genome_params, genome_static_params = policy.compile_genome(genome1)
        population_params, population_static_params = policy.compile_population([genome1, genome2])

        def index_dict_values(data_dict: Dict[str, jax.Array], index: int) -> Dict[str, jax.Array]:
            return {key: value[index] for key, value in data_dict.items()}

        params1 = index_dict_values(population_params, 0)
        static_params1 = index_dict_values(population_static_params, 0)

        policy = NEATPolicy()
        inputs = jnp.array([[0.5, 0.5, 0.5]])
        outputs1 = policy.apply(params1, static_params1, inputs)
        genome_outputs = policy.apply(genome_params, genome_static_params, inputs)

        assert jnp.all(outputs1 == genome_outputs)

    def test_vectorization(self, genome1: NEATGenome, genome2: NEATGenome):
        """Test the vectorization of the policy."""
        policy = NEATPolicy()
        diff_params, static_params = policy.compile_population([genome1, genome2])

        inputs = jnp.array([[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]])

        def single_apply(inp, idx):
            # Extract parameters for this specific genome
            diff_p = {key: value[idx] for key, value in diff_params.items()}
            static_p = {key: value[idx] for key, value in static_params.items()}
            return policy.apply(diff_p, static_p, inp)

        outputs = jax.vmap(single_apply, in_axes=(0, 0))(inputs, jnp.arange(2))

        genome_params1, genome_static_params1 = policy.compile_genome(genome1)
        genome_params2, genome_static_params2 = policy.compile_genome(genome2)
        outputs1 = policy.apply(genome_params1, genome_static_params1, inputs[0:1])
        outputs2 = policy.apply(genome_params2, genome_static_params2, inputs[1:2])

        assert jnp.all(outputs[0] == outputs1)
        assert jnp.all(outputs[1] == outputs2)

    def test_vectorization_with_multiple_outputs(self, fully_connected_genome: NEATGenome):
        """Test vectorization with multiple outputs."""
        policy = NEATPolicy()
        diff_params, static_params = policy.compile_population([fully_connected_genome, fully_connected_genome])

        inputs = jnp.array([[0.5, 0.5], [0.2, 0.3]])

        def single_apply(inp, idx):
            diff_p = {key: value[idx] for key, value in diff_params.items()}
            static_p = {key: value[idx] for key, value in static_params.items()}
            return policy.apply(diff_p, static_p, inp)

        outputs = jax.vmap(single_apply, in_axes=(0, 0))(inputs, jnp.arange(2))

        assert outputs.shape == (2, 2)
