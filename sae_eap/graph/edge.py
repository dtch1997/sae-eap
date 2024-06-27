# from typing import Literal

from dataclasses import dataclass
from sae_eap.graph.node import Node, SrcNode, DestNode

EdgeName = str
# EdgeType = Literal["q", "k", "v", "na"]
# # "q", "k", "v" correspond to the query, key, and value edges to a child attention node.
# # "na" corresponds to an edge that does not have an attention node as a child.


@dataclass(eq=True, frozen=True)
class Edge:
    """Base class to represent an edge in a graph."""

    parent: Node
    child: Node

    @property
    def name(self) -> str:
        """The name of the edge."""
        return f"{self.parent}->{self.child}"

    def __repr__(self):
        return f"Edge({self.name})"

    """ Syntactic sugar """

    @property
    def src(self) -> Node:
        return self.parent

    @property
    def dest(self) -> Node:
        return self.child


@dataclass(eq=True, frozen=True)
class TensorEdge(Edge):
    """An edge connecting two tensors in the model's computational graph."""

    parent: SrcNode
    child: DestNode

    @property
    def src(self) -> SrcNode:
        return self.parent

    @property
    def dest(self) -> DestNode:
        return self.child
