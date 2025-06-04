import jax
import jax.numpy as jnp
import optax
import pytest

from neat.algo.genome import ConnectionGene, NEATGenome, NodeGene, NodeType
from neat.algo.network import Network


class TestNetwork(object):
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

    def test_compile_network(self, genome: NEATGenome):
        """Test the network compilation."""
        network = Network(genome)
        compiled_network = network.compile_network()

        assert "weight_matrix" in compiled_network
        assert "node_ids" in compiled_network
        assert "node_to_idx" in compiled_network
        assert "node_types" in compiled_network
        assert "activation_funcs" in compiled_network
        assert "aggregation_funcs" in compiled_network

        # Check weight matrix dimensions
        num_nodes = len(genome.nodes)
        assert compiled_network["weight_matrix"].shape == (num_nodes, num_nodes)

        # Check node IDs and types
        assert len(compiled_network["node_ids"]) == num_nodes
        assert len(compiled_network["node_types"]) == num_nodes

    def test_forward_pass(self, genome: NEATGenome):
        """Test the forward pass through the network."""
        network = Network(genome)
        inputs = jnp.array([[0.5, 0.5, 0.5]])

        params = {"weight_matrix": network.compiled_network["weight_matrix"]}
        outputs = network.apply(params, inputs)

        assert jnp.all(outputs == 0.75)

    def test_train(self, genome: NEATGenome):
        """Test the training process of the network."""
        network = Network(genome)
        inputs = jnp.array([[0.5, 0.5, 0.5]])
        targets = jnp.array([[1.0]])

        params = {"weight_matrix": network.compiled_network["weight_matrix"]}
        tx = optax.adam(learning_rate=0.01)
        opt_state = tx.init(params)

        def loss_fn(predictions, targets):
            return jnp.mean((predictions - targets) ** 2)

        def model_loss_for_grad(model_params, obs_batch, labels_batch):
            predictions = network.apply(model_params, obs_batch)
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
