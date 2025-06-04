import enum
from dataclasses import dataclass
from typing import Dict, List


class NodeType(enum.Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


class ActivationFunction(enum.Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    SOFTMAX = "softmax"


class AggregationFunction(enum.Enum):
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    PRODUCT = "product"


@dataclass
class NodeGene:
    id: int
    node_type: NodeType
    activation_function: ActivationFunction = ActivationFunction.RELU
    aggregation_function: AggregationFunction = AggregationFunction.SUM

    def __str__(self):
        return f"NodeGene(id={self.id}, type='{self.node_type}', activation='{self.activation_function}', aggregation='{self.aggregation_function}')"


@dataclass
class ConnectionGene:
    in_node_id: int
    out_node_id: int
    weight: float
    enabled: bool

    def __str__(self):
        return (
            f"ConnectionGene(key=({self.in_node_id} -> {self.out_node_id}), "
            f"weight={self.weight:.4f}, enabled={self.enabled}"
        )


@dataclass
class NEATGenome:
    """NEAT genome structure."""

    nodes: Dict[int, NodeGene]
    connections: Dict[int, ConnectionGene]
    fitness: float = 0.0

    def __str__(self):
        connections_str = ", ".join(
            f"{innovation_number} ({conn.in_node_id}->{conn.out_node_id})"
            for innovation_number, conn in self.connections.items()
        )
        return f"NEATGenome({connections_str})"


@dataclass
class NEATState:
    """State for NEAT algorithm."""

    population: List[NEATGenome]
    species: List[List[int]]  # List of species, each containing genome indices
    innovation_counter: int
    node_counter: int
    generation: int
    best_fitness: float
    stagnation_counters: List[int]
