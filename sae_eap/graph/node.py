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
    head_index: int = 0

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


# class LogitNode(Node):
#     """A node corresponding to the output logits of the model."""

#     def __init__(self, n_layers: int):
#         name = "logits"
#         super().__init__(
#             name=name,
#             layer=n_layers - 1,
#             in_hook=f"blocks.{n_layers - 1}.hook_resid_post",
#             out_hook=(),
#         )


# class MLPNode(Node):
#     """A node corresponding to an MLP block in the model."""

#     def __init__(self, layer: int):
#         name = f"m{layer}"
#         super().__init__(
#             name=name,
#             layer=layer,
#             in_hook=f"blocks.{layer}.hook_mlp_in",
#             out_hook=f"blocks.{layer}.hook_mlp_out",
#         )


# class AttentionNode(Node):
#     """A node corresponding to an attention head in the model.

#     NOTE: We have one node per head.
#     """

#     head: int

#     def __init__(self, layer: int, head: int):
#         name = f"a{layer}.h{head}"
#         self.head = head
#         super().__init__(
#             name=name,
#             layer=layer,
#             in_hook=tuple([f"blocks.{layer}.hook_{letter}_input" for letter in "qkv"]),
#             out_hook=f"blocks.{layer}.attn.hook_result",
#         )


# class InputNode(Node):
#     """A node corresponding to the input of the model."""

#     def __init__(self):
#         name = "input"
#         super().__init__(
#             name=name,
#             layer=0,
#             in_hook=(),
#             out_hook="hook_embed",
#         )  # "blocks.0.hook_resid_pre", index)


# # SAE nodes.
# # NOTE: Currently hardcoded to resid_pre


# class SAEReconstructionNode(Node):
#     def __init__(self, layer: int):
#         name = f"sae_recons{layer}"
#         super().__init__(
#             name=name,
#             layer=layer,
#             in_hook=f"blocks.{layer}.hook_resid_pre.hook_sae_input",
#             out_hook=f"blocks.{layer}.hook_resid_pre.hook_sae_recons",
#         )


# class SAEErrorNode(Node):
#     def __init__(self, layer: int):
#         name = f"sae_error{layer}"
#         super().__init__(
#             name=name,
#             layer=layer,
#             in_hook=f"blocks.{layer}.hook_resid_pre.hook_sae_input",
#             out_hook=f"blocks.{layer}.hook_resid_pre.hook_sae_error",
#         )
