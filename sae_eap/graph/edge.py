from typing import Literal

from dataclasses import dataclass
from sae_eap.graph.node import Node
from sae_eap.core.types import HookName

EdgeName = str
EdgeType = Literal["q", "k", "v", "na"]
# "q", "k", "v" correspond to the query, key, and value edges to a child attention node.
# "na" corresponds to an edge that does not have an attention node as a child.


def attn_edge_types() -> list[EdgeType]:
    """Return the edge types corresponding to the query, key, and value edges."""
    return ["q", "k", "v"]


@dataclass
class Edge:
    """Base class to represent an edge in the computational graph.

    Attributes:
        parent: The parent node of the edge.
        child: The child node of the edge.
        type: The type of the edge.
    """

    parent: Node
    child: Node
    type: EdgeType

    @property
    def name(self) -> str:
        """The name of the edge."""
        if self.type == "na":
            return f"{self.parent.name}->{self.child.name}"
        else:
            return f"{self.parent.name}->{self.child.name}<{self.type}>"

    @property
    def child_hook_name(self) -> HookName:
        """The TransformerLens hook name for the child node."""
        if self.type == "na":
            in_hook = self.child.in_hook
            assert isinstance(in_hook, HookName)
            return in_hook
        else:
            return f"blocks.{self.child.layer}.hook_{self.type}_input"

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"Edge({self.name})"

    def __hash__(self):
        return hash(self.name)
