import logging
import math
import random
from collections import defaultdict
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
    NEATSpecies,
    NEATState,
    NodeGene,
    NodeType,
)
from neat.policy import NEATPolicy


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
        prob_replace_weights: float = 0.1,
        prob_perturb_weights: float = 0.8,
        max_stagnation: int = 15,
        survival_threshold: float = 0.2,
        interspecies_mating_rate: float = 0.001,
        activation_function: ActivationFunction = ActivationFunction.IDENTITY,
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
            prob_replace_weights: Probability of weight mutation
            prob_perturb_weights: Probability of weight perturbation
            max_stagnation: Maximum generations without improvement before species elimination
            survival_threshold: Fraction of each species allowed to reproduce
            interspecies_mating_rate: Rate of interspecies mating
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
        self.prob_replace_weights = prob_replace_weights
        self.prob_perturb_weights = prob_perturb_weights
        self.max_stagnation = max_stagnation
        self.survival_threshold = survival_threshold
        self.interspecies_mating_rate = interspecies_mating_rate

        self.activation_function = activation_function

        self.rand_key = jax.random.PRNGKey(seed)

        # Initialize innovation tracking
        self.global_innovations = {}  # Maps (in_node, out_node) -> innovation_number
        self.policy = NEATPolicy()

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
        species_counter = 0
        species = {
            species_counter: NEATSpecies(
                id=species_counter,
                species_indices=list(range(self.pop_size)),
                best_fitness=float("-inf"),
                stagnation_counter=0,
            )
        }
        self.neat_state = NEATState(
            population=population,
            species=species,
            innovation_counter=innovation_counter,
            node_counter=node_counter,
            species_counter=species_counter,
            generation=0,
        )

    def _adjust_fitness_with_explicit_sharing(self):
        """Implement explicit fitness sharing as described in the paper."""
        for species in self.neat_state.species.values():
            # Adjust fitness using explicit sharing formula
            for idx in species.species_indices:
                genome = self.neat_state.population[idx]
                genome.fitness = genome.fitness / len(species.species_indices)

    def _update_stagnation_counters(self):
        """Update stagnation counters for each species."""
        for i, species in self.neat_state.species.items():
            species_indices = species.species_indices

            # Find best fitness in this species
            species_best = max(self.neat_state.population[idx].fitness for idx in species_indices)

            if species_best > species.best_fitness:
                # Species improved
                species.stagnation_counter = 0
                species.best_fitness = species_best
            else:
                # Species did not improve
                species.stagnation_counter += 1

    def _remove_stagnant_species(self):
        """Remove species that have stagnated for too long."""
        stagnant_species = []
        for i, species in self.neat_state.species.items():
            if species.stagnation_counter >= self.max_stagnation:
                stagnant_species.append(i)

        # Only remove stagnant species if we have more than one species
        if len(self.neat_state.species) > len(stagnant_species):
            for i in stagnant_species:
                self.logger.info(f"Removing stagnant species {i}")
                del self.neat_state.species[i]
        else:
            # Keep the best performing stagnant species
            if stagnant_species:
                best_species_id = max(stagnant_species, key=lambda x: self.neat_state.species[x].best_fitness)
                stagnant_species.remove(best_species_id)
                for i in stagnant_species:
                    del self.neat_state.species[i]

    def _choose_species_representatives(self) -> List[NEATGenome]:
        """Select representatives for each species."""
        representatives = {}
        for i, species in self.neat_state.species.items():
            # Each existing species is represented by a random genome inside
            # the species from the previous generation.
            species_indices = species.species_indices
            self.rand_key, key = jax.random.split(self.rand_key)
            parent_idx = int(jax.random.uniform(key) * len(species_indices))
            genome = self.neat_state.population[species_indices[parent_idx]]
            representatives[i] = genome

        return representatives

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

            # If either parent has disabled gene, 75% chance it stays disabled
            if not dominant.connections[innov].enabled or not recessive.connections[innov].enabled:
                if jax.random.uniform(init_key) < 0.75:
                    enabled = False

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

        # Choose a random activation function for the new node
        activation_fn = random.choice(list(ActivationFunction))
        new_nodes[new_node_id] = NodeGene(new_node_id, NodeType.HIDDEN, activation_function=activation_fn)

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
        for conn_key in new_connections:
            if jax.random.uniform(key1) < self.prob_perturb_weights:
                # Perturb weight
                new_connections[conn_key].weight += jax.random.normal(key1) * 0.1
            elif jax.random.uniform(key1) < self.prob_replace_weights + self.prob_perturb_weights:
                # Mutate weight
                new_connections[conn_key].weight = jax.random.normal(key1)

        # Add connection mutation
        if jax.random.uniform(key2) < self.prob_add_connection:
            new_connections = self._mutate_add_connection(new_nodes, new_connections)

        # Add node mutation
        if jax.random.uniform(key3) < self.prob_add_node and new_connections:
            new_nodes, new_connections = self._mutate_add_node(new_nodes, new_connections)

        return NEATGenome(nodes=new_nodes, connections=new_connections)

    def _reproduce_species(self) -> List[NEATGenome]:
        """Reproduce each species to create the next generation."""
        # TODO: The interspecies mating rate was 0.001

        # Calculate total adjusted fitness
        species_fitnesses = [
            sum(self.neat_state.population[idx].fitness for idx in species.species_indices)
            for species in self.neat_state.species.values()
        ]
        total_adjusted_fitness = sum(species_fitnesses)
        assert total_adjusted_fitness > 0, "Total adjusted fitness must be greater than zero."

        # Calculate raw offspring counts (float)
        num_species = len(self.neat_state.species)
        raw_offspring_counts = [
            1 + (sf / total_adjusted_fitness) * (self.pop_size - num_species) for sf in species_fitnesses
        ]

        # Take floor of each and track remainders
        offspring_counts = [math.floor(x) for x in raw_offspring_counts]
        remainders = [x - math.floor(x) for x in raw_offspring_counts]

        # Calculate remaining slots after assigning minimum offspring
        remaining = self.pop_size - int(sum(offspring_counts))
        if remaining > 0:
            sorted_indices = sorted(range(len(remainders)), key=lambda i: remainders[i], reverse=True)
            for i in range(remaining):
                offspring_counts[sorted_indices[i]] += 1

        new_population = []
        for species, offspring_count in zip(self.neat_state.species.values(), offspring_counts):
            species_indices = species.species_indices

            # The champion of each species with more than five networks
            # was copied into the next generation unchanged.
            if len(species_indices) > 5:
                new_population.append(
                    deepcopy(
                        max(
                            (self.neat_state.population[idx] for idx in species_indices),
                            key=lambda genome: genome.fitness,
                        )
                    )
                )
                offspring_count -= 1

            # Sort species by fitness
            species_genomes = [(idx, self.neat_state.population[idx]) for idx in species_indices]
            species_genomes.sort(key=lambda x: x[1].fitness, reverse=True)

            # Reproduce remaining offspring
            survival_count = max(1, int(len(species_genomes) * self.survival_threshold))
            breeding_pool = [genome for _, genome in species_genomes[:survival_count]]

            # In each generation, 25% of offspring resulted from mutation without crossover.
            no_crossover_count = int(offspring_count * 0.25)
            for _ in range(no_crossover_count):
                self.rand_key, subkey = jax.random.split(self.rand_key)
                if breeding_pool:
                    # Asexual reproduction
                    parent_idx = int(jax.random.uniform(subkey) * len(breeding_pool))
                    parent = breeding_pool[parent_idx]
                    offspring = self._mutate_genome(deepcopy(parent))
                    offspring.fitness = 0.0

                    new_population.append(offspring)
                    offspring_count -= 1

            while offspring_count > 0:
                self.rand_key, subkey1, subkey2 = jax.random.split(self.rand_key, 3)

                if len(breeding_pool) < 2:
                    # If breeding pool has less than 2 genomes, just mutate one
                    parent1_idx = int(jax.random.uniform(subkey1) * len(breeding_pool))
                    parent1 = breeding_pool[parent1_idx]
                    offspring = self._mutate_genome(deepcopy(parent1))
                    offspring.fitness = 0.0

                    new_population.append(offspring)
                    offspring_count -= 1
                    continue

                parent1_idx = int(jax.random.uniform(subkey1) * len(breeding_pool))
                parent2_idx = int(jax.random.uniform(subkey2) * len(breeding_pool))

                parent1 = breeding_pool[parent1_idx]
                parent2 = breeding_pool[parent2_idx]

                offspring = self._crossover(parent1, parent2)
                offspring = self._mutate_genome(offspring)
                offspring.fitness = 0.0

                new_population.append(offspring)
                offspring_count -= 1

        assert len(new_population) == self.pop_size

        return new_population[: self.pop_size]

    def _speciate_population(self, species_representatives: Dict[int, NEATSpecies]):
        """
        Divide the population into species based on compatibility distance.

        From the NEAT paper:
        "In each generation, genomes are sequentially placed into species.
        Each existing species is represented by a random genome inside
        the species from the *previous generation*. A given genome *g*
        in the current generation is placed in the first species in which *g*
        is compatible with the representative genome of that species."
        """
        new_species_indices = defaultdict(list)
        unplaced_genomes = []
        for i, genome in enumerate(self.neat_state.population):

            for spec_id, representative in species_representatives.items():
                distance = self._calculate_compatibility_distance(genome, representative)

                if distance < self.compatibility_threshold:
                    # Place genome in this species
                    new_species_indices[spec_id].append(i)
                    break
            else:
                # If not placed in any species, add to unplaced genomes
                unplaced_genomes.append(i)

        for i in unplaced_genomes:
            # If not placed in any species, create a new species
            self.neat_state.species_counter += 1
            spec_id = self.neat_state.species_counter
            new_species = NEATSpecies(
                id=spec_id,
                species_indices=[i],
                best_fitness=float("-inf"),
                stagnation_counter=0,
            )
            self.neat_state.species[spec_id] = new_species
            new_species_indices[spec_id].append(i)

        for spec_id, indices in new_species_indices.items():
            self.neat_state.species[spec_id].species_indices = indices

        for spec_id in list(self.neat_state.species.keys()):
            if spec_id not in new_species_indices:
                # If a species was not updated, it means it has no members left
                del self.neat_state.species[spec_id]

        assert sum(len(species.species_indices) for species in self.neat_state.species.values()) == self.pop_size
        assert len(self.neat_state.species) > 0, "There must be at least one species."

    def _evolve_population(self):
        """Complete evolution step with proper stagnation handling."""
        # Update stagnation counters
        self._update_stagnation_counters()

        # Remove stagnant species
        self._remove_stagnant_species()

        # Calculate explicit fitness sharing
        self._adjust_fitness_with_explicit_sharing()

        # Species representatives
        species_representatives = self._choose_species_representatives()

        # Reproduce species
        new_population = self._reproduce_species()

        # Update population
        self.neat_state.population = new_population

        # Update generation counter
        self.neat_state.generation += 1

        # Re-speciate the new population
        self._speciate_population(species_representatives)

    def ask(self) -> jnp.ndarray:
        """Return current population parameters."""
        return self.policy.compile_population(self.neat_state.population)

    def tell(self, fitness: jnp.ndarray) -> None:
        best_fitness = max(fitness)

        def shift_fitness(rewards):
            min_reward = jnp.min(rewards)
            if min_reward < 0:
                return rewards - min_reward
            return rewards

        fitness = shift_fitness(fitness)

        # Assign fitness scores (keep original fitness values for proper sharing)
        for genome, fit in zip(self.neat_state.population, fitness):
            genome.fitness = float(fit)

        # Evolve population
        self._evolve_population()

        self.logger.info(
            f"Generation {self.neat_state.generation}: "
            f"Best fitness: {best_fitness:.4f}, "
            f"Species count: {len(self.neat_state.species)}, "
            f"Avg stagnation: {np.mean([species.stagnation_counter for species in self.neat_state.species.values()]):.1f}"
        )


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

        species_counter = 0
        species = {
            species_counter: NEATSpecies(
                id=species_counter,
                species_indices=list(range(self.pop_size)),
                best_fitness=float("-inf"),
                stagnation_counter=0,
            )
        }
        self.neat_state = NEATState(
            population=self.population,
            species=species,
            innovation_counter=innovation_counter,
            node_counter=node_counter,
            species_counter=species_counter,
            generation=0,
        )
