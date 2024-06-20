# type: ignore
from typing import Sequence
from sae_eap.graph.graph import TensorGraph
from sae_eap.graph.node import TensorNode, SrcNode, DestNode

# Unique index for each node in the graph.
TensorNodeIndex = int


def build_node_index(nodes: Sequence[TensorNode]) -> dict[TensorNode, TensorNodeIndex]:
    """Build an index for a sequence of nodes."""
    index = {}
    for i, node in enumerate(nodes):
        index[node] = i
    return index


class TensorGraphIndexer:
    """A class to index nodes in a graph."""

    src_index: dict[SrcNode, TensorNodeIndex]
    dest_index: dict[DestNode, TensorNodeIndex]

    def __init__(self, graph: TensorGraph):
        self.build_index(graph)

    def build_index(self, graph: TensorGraph):
        self.src_index = build_node_index(graph.src_nodes)
        self.dest_index = build_node_index(graph.dest_nodes)

    def get_src_index(self, node: TensorNode) -> TensorNodeIndex:
        return self.src_index[node]

    def get_dest_index(self, node: TensorNode) -> TensorNodeIndex:
        return self.dest_index[node]
