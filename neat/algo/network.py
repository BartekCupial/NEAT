from typing import Dict

import jax
import jax.numpy as jnp

from neat.algo.genome import ActivationFunction, AggregationFunction, NEATGenome, NodeType


class Network:
    """Base class for NEAT networks."""

    def __init__(self, genome: NEATGenome):
        self.genome = genome
        self.compiled_network = self.compile_network()

    def compile_network(self) -> Dict:
        """Pre-compile network structure for efficient forward passes."""
        # Build adjacency information
        node_ids = sorted(self.genome.nodes.keys())
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        # Create connection matrix
        num_nodes = len(node_ids)
        weight_matrix = jnp.zeros((num_nodes, num_nodes))

        for conn in self.genome.connections.values():
            if conn.enabled:
                in_idx = node_to_idx[conn.in_node_id]
                out_idx = node_to_idx[conn.out_node_id]
                weight_matrix = weight_matrix.at[out_idx, in_idx].set(conn.weight)

        # Get node types and functions
        node_types = [self.genome.nodes[node_id].node_type for node_id in node_ids]
        activation_funcs = [self.genome.nodes[node_id].activation_function for node_id in node_ids]
        aggregation_funcs = [self.genome.nodes[node_id].aggregation_function for node_id in node_ids]

        return {
            "weight_matrix": weight_matrix,
            "node_ids": node_ids,
            "node_to_idx": node_to_idx,
            "node_types": node_types,
            "activation_funcs": activation_funcs,
            "aggregation_funcs": aggregation_funcs,
        }

    def apply(self, params: Dict, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply the network with given parameters (for training)."""
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = inputs.shape[0]
        num_nodes = len(self.compiled_network["node_ids"])

        # Initialize activations
        activations = jnp.zeros((batch_size, num_nodes))

        # Set input activations
        input_indices = [
            i for i, node_type in enumerate(self.compiled_network["node_types"]) if node_type == NodeType.INPUT
        ]

        for i, idx in enumerate(input_indices):
            if i < inputs.shape[1]:
                activations = activations.at[:, idx].set(inputs[:, i])

        # Use trainable weight matrix from params
        weight_matrix = params["weight_matrix"]

        # Forward propagation (multiple passes to handle recurrence)
        for _ in range(num_nodes):  # Maximum depth
            # Compute weighted inputs for all nodes
            weighted_inputs = jnp.dot(activations, weight_matrix.T)

            # Update non-input nodes
            for i, (node_type, act_func) in enumerate(
                zip(self.compiled_network["node_types"], self.compiled_network["activation_funcs"])
            ):
                if node_type != NodeType.INPUT:
                    # Apply activation function
                    if act_func == ActivationFunction.SIGMOID:
                        new_activation = jax.nn.sigmoid(weighted_inputs[:, i])
                    elif act_func == ActivationFunction.TANH:
                        new_activation = jnp.tanh(weighted_inputs[:, i])
                    elif act_func == ActivationFunction.RELU:
                        new_activation = jax.nn.relu(weighted_inputs[:, i])
                    elif act_func == ActivationFunction.SOFTMAX:
                        new_activation = jax.nn.softmax(weighted_inputs[:, i])
                    else:
                        new_activation = jax.nn.relu(weighted_inputs[:, i])

                    activations = activations.at[:, i].set(new_activation)

        # Extract outputs
        output_indices = [
            i for i, node_type in enumerate(self.compiled_network["node_types"]) if node_type == NodeType.OUTPUT
        ]

        outputs = activations[:, output_indices]

        if squeeze_output:
            outputs = outputs.squeeze(0)

        return outputs
