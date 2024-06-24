from typing import Sequence

from sae_eap.graph.graph import TensorGraph
from sae_eap.graph.node import SrcNode, DestNode, AttentionSrcNode, AttentionDestNode
from sae_eap.graph.edge import TensorEdge

from transformer_lens import HookedTransformer, HookedTransformerConfig


def parse_model_or_config(
    model_or_config: HookedTransformer | HookedTransformerConfig | dict,
) -> HookedTransformerConfig:
    if isinstance(model_or_config, HookedTransformer):
        return model_or_config.cfg
    elif isinstance(model_or_config, HookedTransformerConfig):
        return model_or_config
    elif isinstance(model_or_config, dict):
        return HookedTransformerConfig.from_dict(model_or_config)
    else:
        raise ValueError(
            f"model_or_config must be of type HookedTransformer, HookedTransformerConfig, or dict, but got {type(model_or_config)}"
        )


""" Functions to consruct nodes. """


def get_input_node() -> SrcNode:
    """Return the input node for the graph."""
    return SrcNode(name="Input", hook="hook_embed")


def get_output_node(n_layers: int) -> DestNode:
    """Return the output node for the graph."""
    return DestNode(name="Output", hook=f"blocks.{n_layers - 1}.hook_resid_post")


def get_mlp_nodes(layer: int) -> tuple[SrcNode, DestNode]:
    """Return src and dest nodes for the MLP block in a given layer."""
    in_hook = f"blocks.{layer}.hook_mlp_in"
    out_hook = f"blocks.{layer}.hook_mlp_out"
    src_node = SrcNode(name=f"MLP.L{layer}.out", hook=in_hook)
    dest_node = DestNode(name=f"MLP.L{layer}.in", hook=out_hook)
    return src_node, dest_node


def get_attn_nodes(
    layer: int, n_heads: int
) -> tuple[Sequence[SrcNode], Sequence[DestNode]]:
    """Return src and dest nodes for the attention heads in a given layer."""
    letters = "qkv"
    in_hooks = tuple([f"blocks.{layer}.hook_{letter}_input" for letter in letters])
    out_hook = f"blocks.{layer}.attn.hook_result"
    src_nodes = [
        AttentionSrcNode(
            name=f"ATT.L{layer}.H{head_index}.out", hook=out_hook, head_index=head_index
        )
        for head_index in range(n_heads)
    ]
    dest_nodes = [
        AttentionDestNode(
            name=f"ATT.L{layer}.H{head_index}.in_{letter}",
            hook=in_hook,
            head_index=head_index,
        )
        for head_index in range(n_heads)
        for in_hook, letter in zip(in_hooks, letters)
    ]
    return src_nodes, dest_nodes


def add_layer_nodes_and_edges(
    graph: TensorGraph,
    layer: int,
    prev_src_nodes: list[SrcNode],
) -> list[SrcNode]:
    """Add nodes and edges for a single layer."""

    # Add the nodes
    attn_src_nodes, attn_dest_nodes = get_attn_nodes(layer, graph.cfg.n_heads)
    mlp_src_node, mlp_dest_node = get_mlp_nodes(layer)
    all_layer_nodes = attn_src_nodes + attn_dest_nodes + [mlp_src_node, mlp_dest_node]  # type: ignore
    for node in all_layer_nodes:
        graph.add_node(node)

    # Add the edges
    for src_node in prev_src_nodes:
        for dest_node in attn_dest_nodes + [mlp_dest_node]:  # type: ignore
            assert src_node.is_src
            assert dest_node.is_dest
            graph.add_edge(
                TensorEdge(
                    src_node,
                    dest_node,
                )
            )

    # NOTE: Some models have parallel attention and MLP blocks.
    # This means that the attention nodes are not connected to the MLP node.
    # See: https://arxiv.org/abs/2207.02971
    if not graph.cfg.parallel_attn_mlp:
        # The attention nodes are connected to the MLP node
        for attn_src_node in attn_src_nodes:
            graph.add_edge(
                TensorEdge(
                    attn_src_node,
                    mlp_dest_node,
                )
            )
    else:
        # The attention nodes are not connected to the MLP node
        # So we do nothing here
        pass

    # Update prev_src_nodes
    prev_src_nodes += [mlp_src_node]
    prev_src_nodes += attn_src_nodes

    return prev_src_nodes


def build_graph(
    model_or_config: HookedTransformer | HookedTransformerConfig | dict,
) -> TensorGraph:
    """Build a graph from a model or config."""
    cfg = parse_model_or_config(model_or_config)
    graph = TensorGraph(cfg)

    # Add the input node
    input_node = get_input_node()
    graph.add_node(input_node)

    # Add the intermediate nodes
    prev_src_nodes: list[SrcNode] = [input_node]
    for layer in range(graph.cfg.n_layers):
        prev_src_nodes = add_layer_nodes_and_edges(graph, layer, prev_src_nodes)

    # Add the logit node
    logit_node = get_output_node(graph.cfg.n_layers)
    for node in prev_src_nodes:
        edge = TensorEdge(node, logit_node)
        graph.add_edge(edge)

    graph.add_node(logit_node)
    return graph
