import logging
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger

from neat.algo.genome import (
    ActivationFunction,
    AggregationFunction,
    ConnectionGene,
    NEATGenome,
    NEATState,
    NodeGene,
    NodeType,
)
from neat.algo.network import Network


class NEAT(NEAlgorithm):  # Assuming NEAlgorithm interface from EvoJAX
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        pop_size: int = 150,
        compatibility_threshold: float = 3.0,
        c1: float = 1.0,  # Excess genes coefficient
        c2: float = 1.0,  # Disjoint genes coefficient
        c3: float = 0.4,  # Weight difference coefficient
        prob_add_node: float = 0.03,
        prob_add_connection: float = 0.05,
        prob_mutate_weights: float = 0.8,
        prob_mutate_weight_shift: float = 0.9,
        weight_mutation_power: float = 0.5,
        max_stagnation: int = 15,
        elitism: int = 2,
        survival_threshold: float = 0.2,
        seed: int = 0,
        logger: logging.Logger = None,
    ):
        """Initialize NEAT algorithm.

        Args:
            num_inputs: Number of input nodes
            num_outputs: Number of output nodes
            pop_size: Population size
            compatibility_threshold: Threshold for species compatibility
            c1, c2, c3: Coefficients for compatibility distance calculation
            prob_add_node: Probability of adding a node mutation
            prob_add_connection: Probability of adding a connection mutation
            prob_mutate_weights: Probability of weight mutation
            prob_mutate_weight_shift: Probability of weight shift vs reset
            weight_mutation_power: Standard deviation for weight mutations
            max_stagnation: Maximum generations without improvement before species elimination
            elitism: Number of best genomes to preserve
            survival_threshold: Fraction of each species allowed to reproduce
            seed: Random seed
            logger: Logger instance
        """
        # Set up object variables.

        if logger is None:
            self.logger = create_logger(name="NEAT")
        else:
            self.logger = logger

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.pop_size = pop_size
        self.compatibility_threshold = compatibility_threshold
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.prob_add_node = prob_add_node
        self.prob_add_connection = prob_add_connection
        self.prob_mutate_weights = prob_mutate_weights
        self.prob_mutate_weight_shift = prob_mutate_weight_shift
        self.weight_mutation_power = weight_mutation_power
        self.max_stagnation = max_stagnation
        self.elitism = elitism
        self.survival_threshold = survival_threshold

        self.rand_key = jax.random.PRNGKey(seed)

        # Initialize innovation tracking
        self.global_innovations = {}  # Maps (in_node, out_node) -> innovation_number

        # Initialize population
        self._initialize_population()

    def _initialize_population(self):
        """Initialize population with minimal networks."""
        population = []
        innovation_counter = 0
        node_counter = self.num_inputs + self.num_outputs

        # Create nodes
        nodes = {}

        # Input nodes
        for j in range(self.num_inputs):
            nodes[j] = NodeGene(j, NodeType.INPUT)

        # Output nodes
        for j in range(self.num_outputs):
            node_id = self.num_inputs + j
            nodes[node_id] = NodeGene(node_id, NodeType.OUTPUT)

        # Create initial connections (inputs directly to outputs)
        connections = {}

        for inp in range(self.num_inputs):
            for out in range(self.num_outputs):
                out_id = self.num_inputs + out
                subkey, self.rand_key = jax.random.split(self.rand_key)
                weight = jax.random.normal(subkey, ()) * 0.5

                connections[innovation_counter] = ConnectionGene(
                    in_node_id=inp,
                    out_node_id=out_id,
                    weight=float(weight),
                    enabled=True,
                )

                # Track innovation
                self.global_innovations[(inp, out_id)] = innovation_counter
                innovation_counter += 1

        genome = NEATGenome(nodes=nodes, connections=connections)

        for i in range(self.pop_size):
            genome_clone = deepcopy(genome)
            for connection in genome_clone.connections.values():
                subkey, self.rand_key = jax.random.split(self.rand_key)
                connection.weight = float(jax.random.normal(subkey, ()) * 0.5)
            population.append(genome_clone)

        # Initialize species (initially all genomes in one species)
        species = [list(range(self.pop_size))]
        stagnation_counters = [0]

        self.neat_state = NEATState(
            population=population,
            species=species,
            innovation_counter=innovation_counter,
            node_counter=node_counter,
            generation=0,
            best_fitness=float("-inf"),
            stagnation_counters=stagnation_counters,
        )

    def _group_genes(self, genome1: NEATGenome, genome2: NEATGenome) -> Tuple[set, set, set]:
        """Group genes from two genomes by their innovation numbers."""
        innovations1 = set(genome1.connections.keys())
        innovations2 = set(genome2.connections.keys())

        all_innovations = innovations1.union(innovations2)
        matching = innovations1.intersection(innovations2)

        # Excess and disjoint genes
        excess = set()
        disjoint = set()
        for innov in all_innovations:
            if innov not in matching:
                if innov > min(max(innovations1), max(innovations2)):
                    excess.add(innov)
                else:
                    disjoint.add(innov)

        return matching, excess, disjoint

    def _calculate_compatibility_distance(self, genome1: NEATGenome, genome2: NEATGenome) -> float:
        """Calculate compatibility distance between two genomes."""
        matching, excess, disjoint = self._group_genes(genome1, genome2)

        # Calculate weight difference for matching genes
        # If no matching genes, weight difference is zero
        weight_diff = 0.0
        if matching:
            for innov in matching:
                conn1 = genome1.connections.get(innov)
                conn2 = genome2.connections.get(innov)
                if conn1 and conn2:
                    weight_diff += abs(conn1.weight - conn2.weight)
            weight_diff /= len(matching)

        # Normalize by the number of genes
        N = max(len(genome1.connections), len(genome2.connections))

        # From the NEAT paper:
        # "N can be set to 1 if both genomes are small, i.e., consist of fewer than 20 genes"
        if N < 20:
            N = 1

        distance = self.c1 * len(excess) / N + self.c2 * len(disjoint) / N + self.c3 * weight_diff

        return distance

    def _speciate_population(self):
        """
        Divide the population into species based on compatibility distance.

        From the NEAT paper:
        "In each generation, genomes are sequentially placed into species.
        Each existing species is represented by a random genome inside
        the species from the *previous generation*. A given genome *g*
        in the current generation is placed in the first species in which *g*
        is compatible with the representative genome of that species."
        """
        self.rand_key, init_key = jax.random.split(self.rand_key)

        # Species representatives from the previous generation
        new_species = [
            [jax.random.choice(init_key, self.neat_state.population[species])] for species in self.neat_state.species
        ]

        for i, genome in enumerate(self.neat_state.population):
            placed = False

            for j, species in enumerate(new_species):
                distance = self._calculate_compatibility_distance(genome, species[0])

                if distance < self.compatibility_threshold:
                    # Place genome in this species
                    new_species[j].append(i)
                    placed = True
                    break

            # If not placed in any species, create a new species
            if not placed:
                new_species.append([i])

        self.neat_state.species = new_species

        self._update_stagnation_counters()

    def _build_graph(self, connections: Dict[int, ConnectionGene]) -> nx.DiGraph:
        """Build a directed graph from the connections."""
        graph = nx.DiGraph()
        for conn in connections.values():
            graph.add_edge(conn.in_node_id, conn.out_node_id, weight=conn.weight)
        return graph

    def _mutate_add_connection(self, nodes: Dict[int, NodeGene], connections: Dict[int, ConnectionGene]) -> Dict:
        """Mutate the genome by adding a new connection."""
        self.rand_key, key = jax.random.split(self.rand_key)

        node_ids = list(nodes.keys())

        connection_graph = self._build_graph(connections)

        potential_connections = []
        for in_node in node_ids:
            for out_node in node_ids:
                if (
                    nodes[in_node].node_type != NodeType.OUTPUT
                    and nodes[out_node].node_type != NodeType.INPUT
                    and in_node != out_node
                    and (in_node, out_node) not in connection_graph.edges()
                ):
                    # Skip recurrent connections
                    if nx.has_path(connection_graph, out_node, in_node):
                        continue

                    potential_connections.append((in_node, out_node))

        if potential_connections:
            # Select random connection
            idx = int(jax.random.uniform(key) * len(potential_connections))
            in_node, out_node = potential_connections[idx]

            # Get or create innovation number
            conn_key = (in_node, out_node)
            if conn_key in self.global_innovations:
                innovation_num = self.global_innovations[conn_key]
            else:
                innovation_num = self.neat_state.innovation_counter
                self.global_innovations[conn_key] = innovation_num
                self.neat_state.innovation_counter += 1

            # Create new connection
            weight = float(jax.random.normal(key) * 0.5)
            new_connection = ConnectionGene(in_node_id=in_node, out_node_id=out_node, weight=weight, enabled=True)

            new_connections = connections.copy()
            new_connections[innovation_num] = new_connection
            return new_connections

        return connections

    def _mutate_add_node(self, nodes: Dict[int, NodeGene], connections: Dict[int, ConnectionGene]) -> Tuple[Dict, Dict]:
        # Select random connection to split
        self.rand_key, key = jax.random.split(self.rand_key)

        enabled_innovations = [innov for innov, conn in connections.items() if conn.enabled]
        idx = int(jax.random.uniform(key) * len(enabled_innovations))
        enabled_innov = list(enabled_innovations)[idx]
        old_connection: ConnectionGene = connections[enabled_innov]

        # Create new node
        new_node_id = self.neat_state.node_counter
        self.neat_state.node_counter += 1
        new_nodes = nodes.copy()
        new_nodes[new_node_id] = NodeGene(new_node_id, NodeType.HIDDEN)

        # Disable the old connection and create two new connections
        new_connections = connections.copy()
        new_connections[enabled_innov].enabled = False  # Disable the old connection

        # Connection from old input to new node (weight 1.0)
        conn1_key = (old_connection.in_node_id, new_node_id)
        if conn1_key in self.global_innovations:
            innov1 = self.global_innovations[conn1_key]
        else:
            innov1 = self.neat_state.innovation_counter
            self.global_innovations[conn1_key] = innov1
            self.neat_state.innovation_counter += 1

        new_connections[innov1] = ConnectionGene(
            in_node_id=old_connection.in_node_id,
            out_node_id=new_node_id,
            weight=1.0,
            enabled=True,
        )

        # Connection from new node to original output (original weight)
        conn2_key = (new_node_id, old_connection.out_node_id)
        if conn2_key in self.global_innovations:
            innov2 = self.global_innovations[conn2_key]
        else:
            innov2 = self.neat_state.innovation_counter
            self.global_innovations[conn2_key] = innov2
            self.neat_state.innovation_counter += 1

        new_connections[innov2] = ConnectionGene(
            in_node_id=new_node_id,
            out_node_id=old_connection.out_node_id,
            weight=old_connection.weight,
            enabled=True,
        )

        return new_nodes, new_connections

    def _mutate_genome(self, genome: NEATGenome) -> NEATGenome:
        """Apply mutations to a genome."""
        self.rand_key, key1, key2, key3 = jax.random.split(self.rand_key, 4)

        new_nodes = deepcopy(genome.nodes)
        new_connections = deepcopy(genome.connections)

        # Weight mutations
        if jax.random.uniform(key1) < self.prob_mutate_weights:
            for conn_key in new_connections:
                if jax.random.uniform(key1) < 0.9:  # 90% of connections get mutated
                    if jax.random.uniform(key1) < self.prob_mutate_weight_shift:
                        # Shift weight
                        new_connections[conn_key].weight += jax.random.normal(key1) * self.weight_mutation_power
                    else:
                        # Replace weight
                        new_connections[conn_key].weight = jax.random.normal(key1) * 0.5

        # Add connection mutation
        if jax.random.uniform(key2) < self.prob_add_connection:
            new_connections = self._mutate_add_connection(new_nodes, new_connections)

        # Add node mutation
        if jax.random.uniform(key3) < self.prob_add_node and new_connections:
            new_nodes, new_connections = self._mutate_add_node(new_nodes, new_connections)

        return NEATGenome(nodes=new_nodes, connections=new_connections)

    def _evolve_population(self):
        pass

    def _crossover(self, parent1: NEATGenome, parent2: NEATGenome) -> NEATGenome:
        self.rand_key, init_key = jax.random.split(self.rand_key)

        dominant = parent1 if parent1.fitness > parent2.fitness else parent2
        recessive = parent2 if parent1.fitness > parent2.fitness else parent1

        matching, excess, disjoint = self._group_genes(dominant, recessive)

        offspring_nodes = dict()
        offspring_connections = dict()

        for innov in matching:
            # Matching genes are inherited randomly
            if jax.random.uniform(init_key) < 0.5:
                weight = dominant.connections[innov].weight
                enabled = dominant.connections[innov].enabled
            else:
                weight = recessive.connections[innov].weight
                enabled = recessive.connections[innov].enabled

            offspring_connections[innov] = ConnectionGene(
                in_node_id=dominant.connections[innov].in_node_id,
                out_node_id=dominant.connections[innov].out_node_id,
                weight=weight,
                enabled=enabled,
            )

        for innov in excess.union(disjoint):
            if parent1.fitness != parent2.fitness:
                # Excess and disjoint genes are inherited from the dominant parent
                if innov in dominant.connections:
                    offspring_connections[innov] = deepcopy(dominant.connections[innov])
            else:
                # All disjoint and excess genes from both parents are included in the offspring
                if innov in dominant.connections:
                    offspring_connections[innov] = deepcopy(dominant.connections[innov])

                elif innov in recessive.connections:
                    offspring_connections[innov] = deepcopy(recessive.connections[innov])

        # Create nodes from the connections
        for connection in offspring_connections.values():
            for id in (connection.in_node_id, connection.out_node_id):
                if id not in offspring_nodes:
                    # If the node is not already in the offspring, add it
                    if id in dominant.nodes:
                        offspring_nodes[id] = deepcopy(dominant.nodes[id])
                    elif id in recessive.nodes:
                        offspring_nodes[id] = deepcopy(recessive.nodes[id])
                    else:
                        raise ValueError(f"Node {id} not found in either parent genomes.")

        return NEATGenome(nodes=offspring_nodes, connections=offspring_connections)

    def ask(self) -> jnp.ndarray:
        """Return current population parameters."""
        # Convert genomes to parameter arrays for evaluation
        # TODO:
        pass

    def tell(self, fitness: jnp.ndarray) -> None:
        # 1. Receive fitness scores for the current population.
        # 2. Perform speciation:
        # 3. Update species statistics (e.g., average fitness, stagnation counters).
        # 4. Store the best individuals/species.
        # 5. Update the overall NEAT state.

        # Speciate population
        self._speciate_population()

        # Evolve population
        self._evolve_population()

        self.neat_state.generation += 1

    @property
    def best_params(self) -> jnp.ndarray:
        best_genome = max(self.neat_state.population, key=lambda genome: genome.fitness)

        return Network(best_genome)


class CustomPopulationNEAT(NEAT):
    def __init__(self, population, *args, **kwargs):
        self.population = population
        super().__init__(*args, **kwargs)

    def _initialize_population(self):
        innovation_counter = max(max(genome.connections.keys()) for genome in self.population)
        node_counter = len(set([node_id for genome in self.population for node_id in genome.nodes]))
        self.global_innovations = dict(
            set(
                ((conn.in_node_id, conn.out_node_id), innov)
                for genome in self.population
                for innov, conn in genome.connections.items()
            )
        )

        species = [list(range(self.pop_size))]
        stagnation_counters = [0]
        self.neat_state = NEATState(
            population=self.population,
            species=species,
            innovation_counter=innovation_counter,
            node_counter=node_counter,
            generation=0,
            best_fitness=float("-inf"),
            stagnation_counters=stagnation_counters,
        )
