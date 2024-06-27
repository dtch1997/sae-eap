import torch

from typing import Literal
from sae_eap.core.types import HookPoint, ForwardHook
from sae_eap.graph import TensorGraph, TensorEdge
from sae_eap.data.handler import InputType
from sae_eap.cache import CacheDict


def make_edge_ablation_hook(
    ablate_cache: CacheDict, clean_cache: CacheDict, edge: TensorEdge
) -> ForwardHook:
    """Makes a forward hook for an edge in the model graph.

    - Gets the activation delta of the src node
    - Adds this to the activation of the dest node.
    """
    src_node = edge.parent
    dest_node = edge.child

    def hook_fn(activations: torch.Tensor, hook: HookPoint):
        assert (
            hook.name == dest_node.hook
        ), f"Expected hook name {dest_node.hook}, got {hook.name}"

        # Get the activation delta of the src nodes
        model_ablate_act = ablate_cache[src_node.hook]
        model_clean_act = clean_cache[src_node.hook]
        src_ablate_act = src_node.get_act(model_ablate_act)
        src_store_act = src_node.get_act(model_clean_act)
        src_act_delta = src_ablate_act - src_store_act  # TODO: check sign.

        # Set the activation based on activation of the dest nodes
        model_dest_act = activations
        # First, read the dest node's act from the model
        dest_act = dest_node.get_act(model_dest_act)
        # Then, compute the new dest act by adding the src act delta
        new_dest_act = dest_act + src_act_delta
        # Finally, write the act back to the model act
        dest_node.set_act(model_dest_act, new_dest_act)
        return activations

    return ForwardHook(hook_name=dest_node.hook, hook_fn=hook_fn)


def is_subgraph(circuit_graph: TensorGraph, model_graph: TensorGraph) -> bool:
    """Check if the circuit graph is a subgraph of the model graph."""
    for node in circuit_graph.nodes:
        if not model_graph.has_node(node):
            return False
    for edge in circuit_graph.edges:
        if not model_graph.has_edge(edge):
            return False
    return True


AblateSetting = Literal["noising", "denoising"]


def make_edge_ablate_hooks(
    circuit_graph: TensorGraph,
    model_graph: TensorGraph,
    clean_cache: CacheDict,
    ablate_cache: CacheDict,
) -> list[ForwardHook]:
    # TODO: Write compliant docstring
    """Make edge ablation hooks for all edges not in the circuit graph.

    Parameters:
        circuit_graph: The computation graph of the circuit
        model_graph: The full computation graph of the model
        store_cache: The model's ongoing computation during ablation
        ablate_cache: Pre-computed activations which are used for ablation
    """
    assert is_subgraph(
        circuit_graph, model_graph
    ), "Circuit graph is not a subgraph of the model graph."

    fwd_hooks = []
    for edge in model_graph.edges:
        if circuit_graph.has_edge(edge):
            continue
        fwd_hook = make_edge_ablation_hook(ablate_cache, clean_cache, edge)
        fwd_hooks.append(fwd_hook)

    return fwd_hooks


def get_clean_and_ablate_input(setting: AblateSetting) -> tuple[InputType, InputType]:
    # In noising: We run the model on clean input and ablate with corrupt activations
    # In denoising: We run the model on corrupt input and ablate with clean activations
    # Reference: https://arxiv.org/abs/2404.15255

    if setting == "noising":
        clean_input = "clean"
        ablate_input = "corrupt"
    elif setting == "denoising":
        clean_input = "corrupt"
        ablate_input = "clean"
    else:
        raise ValueError(f"Invalid setting: {setting}")
    return clean_input, ablate_input
