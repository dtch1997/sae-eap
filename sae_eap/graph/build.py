from typing import Sequence

from sae_eap.graph.graph import TensorGraph
from sae_eap.graph.node import TensorNode, AttentionNode
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


""" Functions to construct nodes. """


def build_input_node() -> TensorNode:
    """Return the input node for the graph."""
    return TensorNode(name="Input", hook="hook_embed").as_src()


def build_output_node(n_layers: int) -> TensorNode:
    """Return the output node for the graph."""
    return TensorNode(
        name="Output", hook=f"blocks.{n_layers - 1}.hook_resid_post"
    ).as_dest()


def build_mlp_nodes(layer: int) -> tuple[TensorNode, TensorNode]:
    """Return src and dest nodes for the MLP block in a given layer."""
    in_hook = f"blocks.{layer}.hook_mlp_in"
    out_hook = f"blocks.{layer}.hook_mlp_out"
    src_node = TensorNode(name=f"MLP.L{layer}.out", hook=in_hook).as_src()
    dest_node = TensorNode(name=f"MLP.L{layer}.in", hook=out_hook).as_dest()
    return src_node, dest_node


def build_attn_nodes(layer: int, head: int):
    """Return src and dest nodes for one attention head in a given layer."""
    in_hooks = tuple([f"blocks.{layer}.hook_{letter}_input" for letter in "qkv"])
    out_hook = f"blocks.{layer}.attn.hook_result"
    src_node = AttentionNode(
        name=f"ATT.L{layer}.H{head}.out", hook=out_hook, head_index=head
    ).as_src()
    dest_nodes = [
        AttentionNode(
            name=f"ATT.L{layer}.H{head}.in_{letter}", hook=in_hook, head_index=head
        ).as_dest()
        for in_hook, letter in zip(in_hooks, "qkv")
    ]
    return src_node, dest_nodes


def build_layer_attn_nodes(
    layer: int, n_heads: int
) -> tuple[Sequence[TensorNode], Sequence[TensorNode]]:
    """Return src and dest nodes for the attention heads in a given layer."""
    src_nodes = []
    dest_nodes = []
    for head in range(n_heads):
        src_node, attn_dest_nodes = build_attn_nodes(layer, head)
        src_nodes.append(src_node)
        dest_nodes += attn_dest_nodes
    return src_nodes, dest_nodes


def add_layer_nodes_and_edges(
    graph: TensorGraph,
    layer: int,
    prev_src_nodes: list[TensorNode],
) -> list[TensorNode]:
    """Add nodes and edges for a single layer."""

    # Add the nodes
    attn_src_nodes, attn_dest_nodes = build_layer_attn_nodes(layer, graph.cfg.n_heads)
    mlp_src_node, mlp_dest_node = build_mlp_nodes(layer)
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
    input_node = build_input_node()
    graph.add_node(input_node)

    # Add the intermediate nodes
    prev_src_nodes: list[TensorNode] = [input_node]
    for layer in range(graph.cfg.n_layers):
        prev_src_nodes = add_layer_nodes_and_edges(graph, layer, prev_src_nodes)

    # Add the logit node
    logit_node = build_output_node(graph.cfg.n_layers)
    for node in prev_src_nodes:
        edge = TensorEdge(node, logit_node)
        graph.add_edge(edge)

    graph.add_node(logit_node)
    return graph
