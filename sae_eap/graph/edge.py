# from typing import Literal

from dataclasses import dataclass
from sae_eap.graph.node import Node, SrcNode, DestNode

EdgeName = str
# EdgeType = Literal["q", "k", "v", "na"]
# # "q", "k", "v" correspond to the query, key, and value edges to a child attention node.
# # "na" corresponds to an edge that does not have an attention node as a child.


@dataclass
class Edge:
    """Base class to represent an edge in a graph."""

    parent: Node
    child: Node

    @property
    def name(self) -> str:
        """The name of the edge."""
        return f"{self.parent.name}->{self.child.name}"

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"Edge({self.name})"

    def __hash__(self):
        return hash(self.name)


class TensorEdge(Edge):
    """An edge connecting two tensors in the model's computational graph."""

    parent: SrcNode
    child: DestNode
