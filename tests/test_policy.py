import jax
import jax.numpy as jnp
import optax
import pytest

from neat.algo.genome import ConnectionGene, NEATGenome, NodeGene, NodeType
from neat.policy import NEATPolicy


class TestPolicy(object):
    @pytest.fixture
    def genome(self):
        """Create first parent genome for crossover tests."""
        nodes = {
            1: NodeGene(1, NodeType.INPUT),
            2: NodeGene(2, NodeType.INPUT),
            3: NodeGene(3, NodeType.INPUT),
            4: NodeGene(4, NodeType.OUTPUT),
            5: NodeGene(5, NodeType.HIDDEN),
        }
        connections = {
            1: ConnectionGene(1, 4, 0.5, True),
            2: ConnectionGene(2, 4, 0.5, False),
            3: ConnectionGene(3, 4, 0.5, True),
            4: ConnectionGene(2, 5, 0.5, True),
            5: ConnectionGene(5, 4, 0.5, True),
            8: ConnectionGene(1, 5, 0.5, True),
        }
        return NEATGenome(
            nodes=nodes,
            connections=connections,
            fitness=0.1,
        )

    def test_compile_policy(self, genome: NEATGenome):
        """Test the policy compilation."""
        policy = NEATPolicy()
        policy.compile_genome(genome)

    def test_forward_pass(self, genome: NEATGenome):
        """Test the forward pass through the policy."""
        policy = NEATPolicy()
        params, static_params = policy.compile_genome(genome)
        inputs = jnp.array([[0.5, 0.5, 0.5]])

        outputs = policy.apply(params, static_params, inputs)

        assert jnp.all(outputs == 0.75)

    def test_gradients(self, genome: NEATGenome):
        """Test the gradients of the policy."""
        policy = NEATPolicy()
        params, static_params = policy.compile_genome(genome)

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

    def test_train(self, genome: NEATGenome):
        """Test the training process of the policy."""
        policy = NEATPolicy()
        params, static_params = policy.compile_genome(genome)
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
