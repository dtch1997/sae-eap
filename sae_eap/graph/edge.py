# from typing import Literal

from dataclasses import dataclass
from sae_eap.graph.node import Node, TensorNode

EdgeName = str


# TODO: replace with generic type
@dataclass(eq=True, frozen=True)
class Edge:
    """Base class to represent an edge in a graph."""

    parent: Node
    child: Node

    def __repr__(self):
        return f"Edge({self.parent}->{self.child})"

    """ Syntactic sugar """

    @property
    def src(self) -> Node:
        return self.parent

    @property
    def dest(self) -> Node:
        return self.child

    @property
    def name(self) -> EdgeName:
        return f"{self.parent.name}->{self.child.name}"


@dataclass(eq=True, frozen=True)
class TensorEdge(Edge):
    """An edge connecting two tensors in the model's computational graph."""

    parent: TensorNode
    child: TensorNode

    @property
    def src(self) -> TensorNode:
        return self.parent

    @property
    def dest(self) -> TensorNode:
        return self.child
