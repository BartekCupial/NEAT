from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import networkx as nx

from neat.algo.genome import ActivationFunction, AggregationFunction, ConnectionGene, NEATGenome, NodeGene, NodeType


class Network:
    """JAX-compatible NEAT network implementation."""

    def __init__(self, genome: NEATGenome):
        self.genome = genome
        self.compiled_network = self.compile_network()

    def compile_network(self) -> Dict:
        """Compile the genome into a format suitable for JAX computation."""
        # Build NetworkX graph for topological sorting
        graph = nx.DiGraph()

        # Add nodes
        for node_id, node in self.genome.nodes.items():
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
        for conn in self.genome.connections.values():
            if conn.enabled:
                graph.add_edge(conn.in_node_id, conn.out_node_id, weight=conn.weight)
                active_connections.append((conn.in_node_id, conn.out_node_id, conn.weight))

        # Get topological order for evaluation
        try:
            eval_order = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # Handle cycles by using a simple ordering
            eval_order = sorted(self.genome.nodes.keys())

        # Separate nodes by type
        input_nodes = [nid for nid in eval_order if self.genome.nodes[nid].node_type == NodeType.INPUT]
        hidden_nodes = [nid for nid in eval_order if self.genome.nodes[nid].node_type == NodeType.HIDDEN]
        output_nodes = [nid for nid in eval_order if self.genome.nodes[nid].node_type == NodeType.OUTPUT]

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
        biases = jnp.array([getattr(self.genome.nodes[nid], "bias", 0.0) for nid in all_nodes])
        activations = [self.genome.nodes[nid].activation_function for nid in all_nodes]

        # Create enabled mask
        enabled_mask = jnp.zeros((n_nodes, n_nodes), dtype=bool)

        # Set True for enabled connections
        for conn in self.genome.connections.values():
            if conn.enabled and conn.in_node_id in node_to_idx and conn.out_node_id in node_to_idx:
                i = node_to_idx[conn.in_node_id]
                j = node_to_idx[conn.out_node_id]
                enabled_mask = enabled_mask.at[j, i].set(True)

        return {
            "weights": weights,
            "biases": biases,
            "enabled_mask": enabled_mask,
            "activations": activations,
            "input_indices": list(range(len(input_nodes))),
            "hidden_indices": list(range(len(input_nodes), len(input_nodes) + len(hidden_nodes))),
            "output_indices": list(range(len(input_nodes) + len(hidden_nodes), n_nodes)),
            "eval_order": eval_order,
            "node_to_idx": node_to_idx,
            "n_nodes": n_nodes,
        }

    def init(self) -> Dict:
        differentiate_params = {
            "weights": self.compiled_network["weights"],
            "biases": self.compiled_network["biases"],
        }

        static_params = {
            "enabled_mask": self.compiled_network["enabled_mask"],
            "activations": self.compiled_network["activations"],
            "input_indices": self.compiled_network["input_indices"],
            "hidden_indices": self.compiled_network["hidden_indices"],
            "output_indices": self.compiled_network["output_indices"],
            "n_nodes": self.compiled_network["n_nodes"],
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
        n_nodes = static_params["n_nodes"]

        # Initialize node activations
        activations = jnp.zeros((batch_size, n_nodes))

        # Set input values
        input_indices = static_params["input_indices"]
        activations = activations.at[:, input_indices].set(inputs)

        # Process nodes in evaluation order (excluding inputs)
        # Use stop_gradient to prevent gradients through masked-out connections
        enabled_mask = static_params["enabled_mask"]
        raw_weights = diff_params["weights"]
        stop_gradient = jax.lax.stop_gradient(jnp.zeros_like(raw_weights))

        weights = jnp.where(enabled_mask, raw_weights, stop_gradient)
        biases = diff_params["biases"]

        # For each non-input node, compute its activation
        for i in range(len(input_indices), n_nodes):
            # Aggregate inputs from all connected nodes
            node_input = jnp.dot(activations, weights[i, :]) + biases[i]

            # Apply activation function
            activation_fn = static_params["activations"][i]
            activated = self._apply_activation(node_input, activation_fn)
            activations = activations.at[:, i].set(activated)

        # Extract outputs
        output_indices = static_params["output_indices"]
        outputs = activations[:, output_indices]

        if squeeze_output:
            outputs = outputs.squeeze(0)

        return outputs

    def _apply_activation(self, x: jnp.ndarray, activation_fn: ActivationFunction) -> jnp.ndarray:
        """Apply activation function."""
        if activation_fn == ActivationFunction.SIGMOID:
            return jax.nn.sigmoid(x)
        elif activation_fn == ActivationFunction.TANH:
            return jax.nn.tanh(x)
        elif activation_fn == ActivationFunction.RELU:
            return jax.nn.relu(x)
        elif activation_fn == ActivationFunction.IDENTITY:
            return x
        else:
            return x
