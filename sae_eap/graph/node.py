from __future__ import annotations

import torch

from dataclasses import dataclass
from sae_eap.core.types import HookName
from jaxtyping import Float

NodeName = str


@dataclass(frozen=True, eq=True)
class Node:
    """Base class to represent a node in a graph."""

    name: NodeName

    def __repr__(self):
        return f"Node({self.name})"


@dataclass(frozen=True)
class TensorNode(Node):
    """A node corresponding to a tensor in the model's computational graph."""

    hook: HookName

    @property
    def is_src(self) -> bool:
        return False

    @property
    def is_dest(self) -> bool:
        return False

    def __repr__(self):
        return f"TensorNode({self.name}, {self.hook})"

    def get_act(
        self,
        model_act: torch.Tensor,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        return model_act

    def set_act(
        self,
        model_act: torch.Tensor,
        node_act: Float[torch.Tensor, "batch pos d_model"],
    ) -> None:
        model_act[:] = node_act

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

    def __repr__(self):
        return f"SrcNode({self.name}, {self.hook})"


@dataclass(frozen=True)
class DestNode(TensorNode):
    @property
    def is_dest(self) -> bool:
        return True

    def get_grad(
        self,
        grad: torch.Tensor,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        return grad

    def __repr__(self):
        return f"DestNode({self.name}, {self.hook})"


@dataclass(frozen=True)
class AttentionSrcNode(SrcNode):
    """A node corresponding to an attention head in the model's computational graph."""

    head_index: int

    def get_act(
        self, model_act: Float[torch.Tensor, "batch pos n_head d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        return model_act[:, :, self.head_index]

    def set_act(
        self,
        model_act: Float[torch.Tensor, "batch pos n_head d_model"],
        node_act: Float[torch.Tensor, "batch pos d_model"],
    ) -> None:
        model_act[:, :, self.head_index] = node_act

    def __repr__(self):
        return f"AttentionSrcNode({self.name}, {self.hook}, head={self.head_index})"


@dataclass(frozen=True)
class AttentionDestNode(DestNode):
    """A node corresponding to an attention head in the model's computational graph."""

    head_index: int

    def get_grad(
        self, grad: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos n_head d_model"]:
        return grad[:, :, self.head_index]

    def __repr__(self):
        return f"AttentionDestNode({self.name}, {self.hook}, head={self.head_index})"
