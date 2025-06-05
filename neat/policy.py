from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import networkx as nx
from evojax.policy.base import PolicyNetwork

from neat.algo.genome import ActivationFunction, AggregationFunction, ConnectionGene, NEATGenome, NodeGene, NodeType


class NEATPolicy(PolicyNetwork):
    """JAX-compatible NEAT network implementation."""

    def compile_genome(self, genome: NEATGenome) -> Dict:
        """Compile the genome into a format suitable for JAX computation."""
        # Build NetworkX graph for topological sorting
        graph = nx.DiGraph()

        # Add nodes
        for node_id, node in genome.nodes.items():
            graph.add_node(
                node_id,
                **{
                    "type": node.node_type,
                    "activation": node.activation_function,
                    "aggregation": node.aggregation_function,
                    "bias": getattr(node, "bias", 0.0),
                },
            )

        # Add edges (connections)
        active_connections = []
        for conn in genome.connections.values():
            if conn.enabled:
                graph.add_edge(conn.in_node_id, conn.out_node_id, weight=conn.weight)
                active_connections.append((conn.in_node_id, conn.out_node_id, conn.weight))

        # Get topological order for evaluation
        try:
            eval_order = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # Handle cycles by using a simple ordering
            eval_order = sorted(genome.nodes.keys())

        # Separate nodes by type
        input_nodes = [nid for nid in eval_order if genome.nodes[nid].node_type == NodeType.INPUT]
        hidden_nodes = [nid for nid in eval_order if genome.nodes[nid].node_type == NodeType.HIDDEN]
        output_nodes = [nid for nid in eval_order if genome.nodes[nid].node_type == NodeType.OUTPUT]

        # Create connection matrices for efficient computation
        all_nodes = input_nodes + hidden_nodes + output_nodes
        node_to_idx = {nid: idx for idx, nid in enumerate(all_nodes)}

        # Build weight matrix
        n_nodes = len(all_nodes)
        weights = jnp.zeros((n_nodes, n_nodes))

        for in_node, out_node, weight in active_connections:
            if in_node in node_to_idx and out_node in node_to_idx:
                i, j = node_to_idx[in_node], node_to_idx[out_node]
                weights = weights.at[j, i].set(weight)  # j receives from i

        # Extract node properties
        biases = jnp.array([getattr(genome.nodes[nid], "bias", 0.0) for nid in all_nodes])
        activations = jnp.array([genome.nodes[nid].activation_function.value for nid in all_nodes])

        # Create enabled mask
        enabled_mask = jnp.zeros((n_nodes, n_nodes), dtype=bool)

        # Set True for enabled connections
        for conn in genome.connections.values():
            if conn.enabled and conn.in_node_id in node_to_idx and conn.out_node_id in node_to_idx:
                i = node_to_idx[conn.in_node_id]
                j = node_to_idx[conn.out_node_id]
                enabled_mask = enabled_mask.at[j, i].set(True)

        input_indices = jnp.arange(len(input_nodes))
        output_indices = jnp.arange(len(output_nodes)) + len(input_nodes) + len(hidden_nodes)

        differentiate_params = {
            "weights": weights,
            "biases": biases,
        }

        static_params = {
            "enabled_mask": enabled_mask,
            "activations": activations,
            "input_indices": input_indices,
            "output_indices": output_indices,
            "eval_order": eval_order,
        }

        return differentiate_params, static_params

    def compile_population(self, genomes: List[NEATGenome]) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
        # Find maximum dimensions across population
        max_nodes = max(len(genome.nodes) for genome in genomes)
        num_inputs = len([n for n in genomes[0].nodes.values() if n.node_type == NodeType.INPUT])
        num_outputs = len([n for n in genomes[0].nodes.values() if n.node_type == NodeType.OUTPUT])
        batch_size = len(genomes)

        # Initialize batched tensors
        weights = jnp.zeros((batch_size, max_nodes, max_nodes))
        biases = jnp.zeros((batch_size, max_nodes))
        enabled_mask = jnp.zeros((batch_size, max_nodes, max_nodes), dtype=bool)
        activations = jnp.zeros((batch_size, max_nodes), dtype=jnp.int32)

        # Track node ordering for each genome
        input_indices = jnp.zeros((batch_size, num_inputs), dtype=jnp.int32)
        output_indices = jnp.zeros((batch_size, num_outputs), dtype=jnp.int32)

        for i, genome in enumerate(genomes):
            params, static_params = self.compile_genome(genome)

            # Get actual dimensions of this genome
            genome_weights = params["weights"]
            genome_biases = params["biases"]
            genome_enabled_mask = static_params["enabled_mask"]
            genome_input_indices = static_params["input_indices"]
            genome_output_indices = static_params["output_indices"]
            genome_activations = static_params["activations"]

            h, w = genome_weights.shape
            b_len = genome_biases.shape[0]

            # Slice assignment to handle smaller shapes
            weights = weights.at[i, :h, :w].set(genome_weights)
            biases = biases.at[i, :b_len].set(genome_biases)

            enabled_mask = enabled_mask.at[i, :h, :w].set(genome_enabled_mask)
            activations = activations.at[i, :h].set(genome_activations)
            input_indices = input_indices.at[i].set(genome_input_indices)
            output_indices = output_indices.at[i].set(genome_output_indices)

        differentiate_params = {
            "weights": weights,
            "biases": biases,
        }

        static_params = {
            "enabled_mask": enabled_mask,
            "activations": activations,
            "input_indices": input_indices,
            "output_indices": output_indices,
        }

        return differentiate_params, static_params

    def apply(self, diff_params: Dict, static_params: Dict, inputs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network."""

        # Handle batch dimension
        if inputs.ndim == 1:
            inputs = inputs[None, :]  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = inputs.shape[0]

        # Process nodes in evaluation order (excluding inputs)
        # Use stop_gradient to prevent gradients through masked-out connections
        enabled_mask = static_params["enabled_mask"]
        raw_weights = diff_params["weights"]
        stop_gradient = jax.lax.stop_gradient(jnp.zeros_like(raw_weights))

        weights = jnp.where(enabled_mask, raw_weights, stop_gradient)
        biases = diff_params["biases"]

        # Initialize node activations
        activations = jnp.zeros((batch_size, weights.shape[0]))

        # Set input values
        input_indices = static_params["input_indices"]
        expanded_inputs = jnp.zeros((batch_size, weights.shape[0]))
        expanded_inputs = expanded_inputs.at[:, : inputs.shape[1]].set(inputs)
        activations = activations.at[:, input_indices].set(inputs)

        n_inputs = input_indices.shape[0]
        n_nodes = weights.shape[0]

        # Use jax.lax.fori_loop instead of Python for loop
        def process_node(i, activations):
            # Only process non-input nodes
            node_input = jnp.dot(activations, weights[i, :]) + biases[i]
            activation_fn = static_params["activations"][i]
            activated = self._apply_activation(node_input, activation_fn)

            # Conditionally update only if i >= n_inputs
            should_update = i >= n_inputs
            new_activation = jnp.where(should_update, activated, activations[:, i])
            return activations.at[:, i].set(new_activation)

        activations = jax.lax.fori_loop(n_inputs, n_nodes, process_node, activations)

        # Extract outputs
        output_indices = static_params["output_indices"]
        outputs = jnp.take(activations, output_indices, axis=1)

        if squeeze_output:
            outputs = outputs.squeeze(0)

        return outputs

    def _apply_activation(self, x: jnp.ndarray, activation_fn: ActivationFunction) -> jnp.ndarray:
        """Apply activation function using JAX-compatible switch."""

        activation_funcs = [
            lambda x: jax.nn.sigmoid(x),  # SIGMOID (index 0)
            lambda x: jax.nn.tanh(x),  # TANH (index 1)
            lambda x: jax.nn.relu(x),  # RELU (index 2)
            lambda x: jax.nn.softmax(x),  # SOFTMAX (index 3)
            lambda x: x,  # IDENTITY (index 4)
            lambda x: 1.0 / (1.0 + jnp.exp(-4.9 * x)),  # MODIFIED_SIGMOID (index 5)
        ]

        return jax.lax.switch(activation_fn, activation_funcs, x)

    def get_actions(self, t_states, params, p_states):
        observations = t_states.obs

        diff_params, static_params = params

        def single_apply(inp, idx):
            # Extract parameters for this specific genome
            diff_p = {key: value[idx] for key, value in diff_params.items()}
            static_p = {key: value[idx] for key, value in static_params.items()}
            return self.apply(diff_p, static_p, inp)

        actions = jax.vmap(single_apply, in_axes=(0, 0))(observations, jnp.arange(observations.shape[0]))

        return actions, p_states
