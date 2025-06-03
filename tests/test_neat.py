import pytest

from neat.algo import NEAT
from neat.algo.neat import ConnectionGene, CustomPopulationNEAT, NEATGenome, NodeGene, NodeType


class TestNEAT(object):
    @pytest.fixture
    def parent1_genome(self):
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
    def parent2_genome(self):
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
    def neat_instance(self, parent1_genome: NEATGenome, parent2_genome: NEATGenome):
        """Create a NEAT instance for testing."""
        population = [parent1_genome, parent2_genome]
        return CustomPopulationNEAT(population, num_inputs=3, num_outputs=1, pop_size=2, seed=42)

    def test_crossover_different(self, neat_instance: NEAT, parent1_genome: NEATGenome, parent2_genome: NEATGenome):
        """Test NEAT crossover when parent2 has higher fitness."""
        child = neat_instance._crossover(parent1_genome, parent2_genome)

        assert len(child.nodes) == 6
        assert len(child.connections) == 9

        # here for simplicity we only test node_ids, since we will be doing backprop weigths don't matter
        # child should have the same nodes and connections as parent2 (different fitness)
        for node_id in child.nodes:
            assert node_id in parent2_genome.nodes
        for conn_id in child.connections:
            assert conn_id in parent2_genome.connections

    def test_crossover_same(self, neat_instance: NEAT, parent1_genome: NEATGenome, parent2_genome: NEATGenome):
        # make both parents have the same fitness
        parent2_genome.fitness = parent1_genome.fitness = 1.0

        child = neat_instance._crossover(parent1_genome, parent2_genome)

        assert len(child.nodes) == 6
        assert len(child.connections) == 10

        # here for simplicity we only test node_ids, since we will be doing backprop weigths don't matter
        # child should have the same nodes and connections both parents (the same fitness)
        for node_id in child.nodes:
            assert node_id in parent2_genome.nodes or node_id in parent1_genome.nodes
        for conn_id in child.connections:
            assert conn_id in parent2_genome.connections or conn_id in parent1_genome.connections

    def test_mutate_add_connection(self, neat_instance: NEAT, parent1_genome: NEATGenome):
        new_connections = neat_instance._mutate_add_connection(parent1_genome.nodes, parent1_genome.connections)
        assert len(new_connections) == len(parent1_genome.connections) + 1

    def test_mutate_add_node(self, neat_instance: NEAT, parent1_genome: NEATGenome):
        new_nodes, new_connections = neat_instance._mutate_add_node(parent1_genome.nodes, parent1_genome.connections)
        assert len(new_nodes) == len(parent1_genome.nodes) + 1
        assert len(new_connections) == len(parent1_genome.connections) + 2

    def test_mutate_genome(self, neat_instance: NEAT, parent1_genome: NEATGenome):
        """Test NEAT mutation."""
        # since there is no guarantee that mutation will happen, just check if mutate doesn't raise an error
        neat_instance._mutate_genome(parent1_genome)

    def test_calculate_compability_distance(
        self, neat_instance: NEAT, parent1_genome: NEATGenome, parent2_genome: NEATGenome
    ):
        """Test NEAT compatibility distance calculation."""
        distance = neat_instance._calculate_compatibility_distance(parent1_genome, parent2_genome)
        assert isinstance(distance, float)
        # we have total of 5 disjoint or execess genes, all weights are 0.5, so distance should be 5.0
        assert distance == 5.0

    def test_evolve(self):
        pass
