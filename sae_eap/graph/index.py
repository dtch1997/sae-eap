# type: ignore
from typing import Sequence
from sae_eap.graph.graph import TensorGraph
from sae_eap.graph.node import TensorNode

# Unique index for each node in the graph.
TensorNodeIndex = int


def build_node_index(nodes: Sequence[TensorNode]) -> dict[TensorNode, TensorNodeIndex]:
    """Build an index for a sequence of nodes."""
    index = {}
    for i, node in enumerate(nodes):
        index[node] = i
    return index


# TODO: refactor into generic Indexer class that supports any hashable key.
# TODO: instead of combining src and dest in a single index, have two separate indexers.
class TensorGraphIndexer:
    """A class to index nodes in a graph."""

    src_index: dict[TensorNode, TensorNodeIndex]
    dest_index: dict[TensorNode, TensorNodeIndex]

    def __init__(self, graph: TensorGraph):
        self.build_index(graph)

    def build_index(self, graph: TensorGraph):
        self.src_index = build_node_index(graph.src_nodes)
        self.dest_index = build_node_index(graph.dest_nodes)

    def get_src_index(self, node: TensorNode) -> TensorNodeIndex:
        return self.src_index[node]

    def get_dest_index(self, node: TensorNode) -> TensorNodeIndex:
        return self.dest_index[node]
