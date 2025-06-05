import pytest

from neat.algo import NEAT
from neat.algo.neat import CustomPopulationNEAT, NEATGenome


class TestNEAT(object):
    @pytest.fixture
    def neat_instance(self, genome1: NEATGenome, genome2: NEATGenome):
        """Create a NEAT instance for testing."""
        population = [genome1, genome2]
        return CustomPopulationNEAT(population, num_inputs=3, num_outputs=1, pop_size=2, seed=42)

    def test_crossover_different(self, neat_instance: NEAT, genome1: NEATGenome, genome2: NEATGenome):
        """Test NEAT crossover when parent2 has higher fitness."""
        child = neat_instance._crossover(genome1, genome2)

        assert len(child.nodes) == 6
        assert len(child.connections) == 9

        # here for simplicity we only test node_ids, since we will be doing backprop weigths don't matter
        # child should have the same nodes and connections as parent2 (different fitness)
        for node_id in child.nodes:
            assert node_id in genome2.nodes
        for conn_id in child.connections:
            assert conn_id in genome2.connections

    def test_crossover_same(self, neat_instance: NEAT, genome1: NEATGenome, genome2: NEATGenome):
        # make both parents have the same fitness
        genome2.fitness = genome1.fitness = 1.0

        child = neat_instance._crossover(genome1, genome2)

        assert len(child.nodes) == 6
        assert len(child.connections) == 10

        # here for simplicity we only test node_ids, since we will be doing backprop weigths don't matter
        # child should have the same nodes and connections both parents (the same fitness)
        for node_id in child.nodes:
            assert node_id in genome2.nodes or node_id in genome1.nodes
        for conn_id in child.connections:
            assert conn_id in genome2.connections or conn_id in genome1.connections

    def test_mutate_add_connection(self, neat_instance: NEAT, genome1: NEATGenome):
        new_connections = neat_instance._mutate_add_connection(genome1.nodes, genome1.connections)
        assert len(new_connections) == len(genome1.connections) + 1

    def test_mutate_add_node(self, neat_instance: NEAT, genome1: NEATGenome):
        new_nodes, new_connections = neat_instance._mutate_add_node(genome1.nodes, genome1.connections)
        assert len(new_nodes) == len(genome1.nodes) + 1
        assert len(new_connections) == len(genome1.connections) + 2

    def test_mutate_genome(self, neat_instance: NEAT, genome1: NEATGenome):
        """Test NEAT mutation."""
        # since there is no guarantee that mutation will happen, just check if mutate doesn't raise an error
        neat_instance._mutate_genome(genome1)

    def test_calculate_compability_distance(self, neat_instance: NEAT, genome1: NEATGenome, genome2: NEATGenome):
        """Test NEAT compatibility distance calculation."""
        distance = neat_instance._calculate_compatibility_distance(genome1, genome2)
        assert isinstance(distance, float)
        # we have total of 5 disjoint or execess genes, all weights are 0.5, so distance should be 5.0
        assert distance == 5.0

    def test_evolve(self):
        pass
