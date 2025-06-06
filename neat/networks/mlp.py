from itertools import pairwise

import jax
import numpy as np

from neat.algo.genome import ActivationFunction, ConnectionGene, NEATGenome, NodeGene, NodeType


def mlp(layers=[2, 4, 2], activation_function=ActivationFunction.TANH, key=jax.random.PRNGKey(0)):
    """Create a fully connected genome for the XOR task."""
    node_types = [NodeType.INPUT]
    node_types.extend([NodeType.HIDDEN for _ in range(len(layers) - 2)])
    node_types.append(NodeType.OUTPUT)

    nodes = {}
    node_idx = 0
    for layer, node_type in zip(layers, node_types):
        for _ in range(layer):
            nodes[node_idx] = NodeGene(node_idx, node_type, activation_function=activation_function)
            node_idx += 1

    node_cumsum = [0] + np.cumsum(np.array(layers)).tolist()
    connections = {}
    conn_idx = 0
    for start_idx, end_idx in pairwise(node_cumsum):
        # Get the next layer's start and end indices
        next_layer_start = end_idx
        next_layer_end = (
            node_cumsum[node_cumsum.index(end_idx) + 1]
            if node_cumsum.index(end_idx) + 1 < len(node_cumsum)
            else end_idx
        )

        for i in range(start_idx, end_idx):
            for j in range(next_layer_start, next_layer_end):
                key, subkey = jax.random.split(key)
                weight = jax.random.uniform(subkey)
                connections[conn_idx] = ConnectionGene(i, j, weight, True)
                conn_idx += 1

    return NEATGenome(
        nodes=nodes,
        connections=connections,
        fitness=0.0,
    )
