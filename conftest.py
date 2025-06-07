import jax
import pytest

from neat.algo.neat import ActivationFunction, ConnectionGene, NEATGenome, NodeGene, NodeType
from neat.networks.mlp import mlp


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
    return mlp(
        layers=[2, 4, 2],
        activation_function=ActivationFunction.TANH,
        last_activation_function=ActivationFunction.IDENTITY,
        key=jax.random.PRNGKey(0),
    )
