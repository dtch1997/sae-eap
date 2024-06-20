from __future__ import annotations
from dataclasses import dataclass
from sae_eap.core.types import HookName

NodeName = str


@dataclass(frozen=True)
class Node:
    """Base class to represent a node in a graph."""

    name: NodeName

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"Node({self.name})"


@dataclass(frozen=True)
class TensorNode(Node):
    """A node corresponding to a tensor in the model's computational graph."""

    hook: HookName
    head_index: int | None = None

    @property
    def is_src(self) -> bool:
        return False

    @property
    def is_dest(self) -> bool:
        return False

    """ Syntactic sugar for deciding what tensors to store. """

    @property
    def requires_grad(self) -> bool:
        """Indicates whether we need to keep track of gradients at this hook."""
        return self.is_dest

    @property
    def requires_act(self) -> bool:
        """Indicates whether we need to keep track of activations at this hook."""
        return self.is_src


@dataclass(frozen=True)
class SrcNode(TensorNode):
    @property
    def is_src(self) -> bool:
        return True


@dataclass(frozen=True)
class DestNode(TensorNode):
    @property
    def is_dest(self) -> bool:
        return True
