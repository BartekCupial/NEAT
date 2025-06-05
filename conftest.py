import jax
import pytest

from neat.algo.neat import ActivationFunction, ConnectionGene, NEATGenome, NodeGene, NodeType


@pytest.fixture
def genome1():
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


@pytest.fixture
def genome2():
    """Create second parent genome for crossover tests."""
    nodes = {
        1: NodeGene(1, NodeType.INPUT),
        2: NodeGene(2, NodeType.INPUT),
        3: NodeGene(3, NodeType.INPUT),
        4: NodeGene(4, NodeType.OUTPUT),
        5: NodeGene(5, NodeType.HIDDEN),
        6: NodeGene(6, NodeType.HIDDEN),
    }
    connections = {
        1: ConnectionGene(1, 4, 0.5, True),
        2: ConnectionGene(2, 4, 0.5, False),
        3: ConnectionGene(3, 4, 0.5, True),
        4: ConnectionGene(2, 5, 0.5, True),
        5: ConnectionGene(5, 4, 0.5, False),
        6: ConnectionGene(5, 6, 0.5, True),
        7: ConnectionGene(6, 4, 0.5, True),
        9: ConnectionGene(3, 5, 0.5, True),
        10: ConnectionGene(1, 6, 0.5, True),
    }
    return NEATGenome(
        nodes=nodes,
        connections=connections,
        fitness=0.2,
    )


@pytest.fixture
def fully_connected_genome():
    """Create a fully connected genome for the XOR task."""
    input_nodes = 2
    hidden_nodes = 4
    output_nodes = 2

    nodes = {}
    for i in range(1, input_nodes + 1):
        nodes[i] = NodeGene(i, NodeType.INPUT, activation_function=ActivationFunction.TANH)

    for i in range(input_nodes + 1, input_nodes + hidden_nodes + 1):
        nodes[i] = NodeGene(i, NodeType.HIDDEN, activation_function=ActivationFunction.TANH)

    for i in range(input_nodes + hidden_nodes + 1, input_nodes + hidden_nodes + output_nodes + 1):
        nodes[i] = NodeGene(i, NodeType.OUTPUT)

    connections = {}
    # connections from input nodes to hidden nodes
    conn_idx = 1
    for i in range(1, input_nodes + 1):
        for j in range(input_nodes + 1, input_nodes + hidden_nodes + 1):
            weight = jax.random.uniform(jax.random.PRNGKey(conn_idx))
            connections[conn_idx] = ConnectionGene(i, j, weight, True)
            conn_idx += 1

    # connections from hidden nodes to output nodes
    for i in range(input_nodes + 1, input_nodes + hidden_nodes + 1):
        for j in range(input_nodes + hidden_nodes + 1, input_nodes + hidden_nodes + output_nodes + 1):
            weight = jax.random.uniform(jax.random.PRNGKey(conn_idx))
            connections[conn_idx] = ConnectionGene(i, j, weight, True)
            conn_idx += 1

    return NEATGenome(
        nodes=nodes,
        connections=connections,
        fitness=0.0,
    )
