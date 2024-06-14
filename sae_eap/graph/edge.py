from typing import Literal
from sae_eap.graph.node import Node
from dataclasses import dataclass

EdgeType = Literal["q", "k", "v", "na"]
# "q", "k", "v" correspond to the query, key, and value edges to a child attention node.
# "na" corresponds to an edge that does not have an attention node as a child.


@dataclass
class Edge:
    parent: Node
    child: Node
    type: EdgeType

    @property
    def name(self):
        if self.type == "na":
            return f"{self.parent.name}->{self.child.name}"
        else:
            return f"{self.parent.name}->{self.child.name}<{self.type}>"

    @property
    def hook(self):
        if self.type == "na":
            return self.child.in_hook
        else:
            return f"blocks.{self.child.layer}.hook_{self.type}_input"

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"Edge({self.name})"

    def __hash__(self):
        return hash(self.name)
