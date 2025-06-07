import io
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image

from neat.algo.genome import NEATGenome, NodeType


def save_genome(genome: NEATGenome, file_path: str) -> None:
    """Save a genome to a file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(genome, f)


def render_genome(genome: NEATGenome) -> np.ndarray:
    # Build NetworkX graph for topological sorting
    G = nx.DiGraph()

    # Add nodes
    for node_id, node in genome.nodes.items():
        G.add_node(
            node_id,
            **{
                "type": node.node_type,
                "activation": node.activation_function,
                "aggregation": node.aggregation_function,
                "bias": getattr(node, "bias", 0.0),
            },
        )

    for connection in genome.connections.values():
        if connection.enabled:
            G.add_edge(connection.in_node_id, connection.out_node_id, weight=connection.weight)

    pos = {}

    input_nodes = [node_id for node_id, node in genome.nodes.items() if node.node_type == NodeType.INPUT]
    output_nodes = [node_id for node_id, node in genome.nodes.items() if node.node_type == NodeType.OUTPUT]

    # Calculate layer depth for each node
    layer_depth = {}

    try:
        sorted_nodes = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print("Error: The graph has cycles and cannot be topologically sorted for layered layout.")
        pos = nx.spring_layout(G)
    else:
        # Assign layer depth for input nodes
        for node in input_nodes:
            layer_depth[node] = 0

        # Assign layer depth for other nodes based on predecessors
        for node in sorted_nodes:
            if node not in layer_depth:
                preds = list(G.predecessors(node))
                if not preds:
                    layer_depth[node] = 0
                else:
                    layer_depth[node] = max(layer_depth[p] for p in preds) + 1

        max_depth = max(layer_depth.values()) if layer_depth else 0

        # Assign output nodes to the final layer
        for node in output_nodes:
            if max_depth > 0:  # Only override if there are hidden layers
                layer_depth[node] = max_depth

        # Recalculate max_depth in case output nodes changed it
        max_depth = max(layer_depth.values()) if layer_depth else 0

        # If all nodes are in one layer, prevent division by zero
        if max_depth == 0:
            max_depth = 1

        # Position nodes based on layer depth
        pos = {}
        nodes_by_layer = {}
        for node, depth in layer_depth.items():
            if depth not in nodes_by_layer:
                nodes_by_layer[depth] = []
            nodes_by_layer[depth].append(node)

        for depth, nodes_in_layer in nodes_by_layer.items():
            y = depth / max_depth
            x_coords = np.linspace(0.05, 0.95, len(nodes_in_layer)) if len(nodes_in_layer) > 1 else [0.5]
            for i, node in enumerate(sorted(nodes_in_layer)):
                pos[node] = (x_coords[i], y)

    plt.figure(figsize=(7, 7))
    node_size = 500  # Define node size as a variable for consistency

    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightgray", node_size=node_size)

    # Draw input and output nodes
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color="skyblue", node_size=node_size)
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color="salmon", node_size=node_size)

    # Draw edges with node_size specified to avoid overlap
    nx.draw_networkx_edges(G, pos, node_size=node_size, arrowstyle="->", arrowsize=10, width=1.5)[2]

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_color="black", font_weight="bold", font_size=8)

    plt.tight_layout()

    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()

    # Convert buffer to PIL Image
    buf.seek(0)
    img = Image.open(buf)

    img_array = np.array(img)

    # Convert to RGB array (removing alpha channel if present)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    return img_array


def save_genome_img(genome: NEATGenome, file_path: str) -> None:
    """Save a genome's topology to a file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    img_array = render_genome(genome)
    img = Image.fromarray(img_array)
    img.save(file_path)
