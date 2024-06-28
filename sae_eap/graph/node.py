from __future__ import annotations

import torch

from dataclasses import dataclass, asdict
from sae_eap.core.types import HookName
from jaxtyping import Float

NodeName = str


@dataclass(frozen=True, eq=True, kw_only=True)
class Node:
    """Base class to represent a node in a graph."""

    name: NodeName
    is_src: bool = False
    is_dest: bool = False

    def __repr__(self):
        return f"Node({self.name})"

    def as_src(self, value: bool = True):
        dict = asdict(self)
        dict["is_src"] = value
        return self.__class__(**dict)

    def as_dest(self, value: bool = True):
        dict = asdict(self)
        dict["is_dest"] = value
        return self.__class__(**dict)


@dataclass(frozen=True, eq=True, kw_only=True)
class TensorNode(Node):
    """A node corresponding to a tensor in the model's computational graph."""

    name: NodeName
    hook: HookName
    is_src: bool = False
    is_dest: bool = False

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

    def get_grad(
        self,
        model_grad: torch.Tensor,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        return model_grad

    def set_grad(
        self,
        model_grad: torch.Tensor,
        node_grad: Float[torch.Tensor, "batch pos d_model"],
    ) -> None:
        model_grad[:] = node_grad


@dataclass(frozen=True, eq=True, kw_only=True)
class AttentionNode(TensorNode):
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

    def get_grad(
        self, model_grad: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos n_head d_model"]:
        return model_grad[:, :, self.head_index]

    def set_grad(
        self,
        model_grad: Float[torch.Tensor, "batch pos n_head d_model"],
        node_grad: Float[torch.Tensor, "batch pos d_model"],
    ) -> None:
        model_grad[:, :, self.head_index] = node_grad

    def __repr__(self):
        return f"AttentionNode({self.name}, {self.hook}, head={self.head_index})"
