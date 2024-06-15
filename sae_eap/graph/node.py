from __future__ import annotations
from dataclasses import dataclass
from sae_eap.core.types import HookName

NodeName = str


@dataclass
class Node:
    """
    Base class to represent a node in the computational graph.

    Attributes:
        name: The name of the node.
        layer: The layer of the node.
        in_hooks: TransformerLens hook(s) for the input(s).
        out_hook: TransformerLens hook for the output.
    """

    name: NodeName
    layer: int
    in_hook: HookName | tuple[HookName, ...]
    out_hook: HookName

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f"Node({self.name})"

    def __hash__(self):
        return hash(self.name)


class LogitNode(Node):
    """A node corresponding to the output logits of the model."""

    def __init__(self, n_layers: int):
        name = "logits"
        super().__init__(
            name=name,
            layer=n_layers - 1,
            in_hook=f"blocks.{n_layers - 1}.hook_resid_post",
            out_hook="",
        )


class MLPNode(Node):
    """A node corresponding to an MLP block in the model."""

    def __init__(self, layer: int):
        name = f"m{layer}"
        super().__init__(
            name=name,
            layer=layer,
            in_hook=f"blocks.{layer}.hook_mlp_in",
            out_hook=f"blocks.{layer}.hook_mlp_out",
        )


class AttentionNode(Node):
    """A node corresponding to an attention head in the model.

    NOTE: We have one node per head.
    """

    head: int

    def __init__(self, layer: int, head: int):
        name = f"a{layer}.h{head}"
        self.head = head
        super().__init__(
            name=name,
            layer=layer,
            in_hook=tuple([f"blocks.{layer}.hook_{letter}_input" for letter in "qkv"]),
            out_hook=f"blocks.{layer}.attn.hook_result",
        )


class InputNode(Node):
    """A node corresponding to the input of the model."""

    def __init__(self):
        name = "input"
        super().__init__(
            name=name,
            layer=0,
            in_hook="",
            out_hook="hook_embed",
        )  # "blocks.0.hook_resid_pre", index)


# SAE nodes.
# NOTE: Currently hardcoded to resid_pre


class SAEReconstructionNode(Node):
    def __init__(self, layer: int):
        name = f"sae_recons{layer}"
        super().__init__(
            name=name,
            layer=layer,
            in_hook=f"blocks.{layer}.hook_resid_pre.hook_sae_input",
            out_hook=f"blocks.{layer}.hook_resid_pre.hook_sae_recons",
        )


class SAEErrorNode(Node):
    def __init__(self, layer: int):
        name = f"sae_error{layer}"
        super().__init__(
            name=name,
            layer=layer,
            in_hook=f"blocks.{layer}.hook_resid_pre.hook_sae_input",
            out_hook=f"blocks.{layer}.hook_resid_pre.hook_sae_error",
        )
