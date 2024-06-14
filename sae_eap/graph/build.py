from sae_eap.graph.graph import Graph
from sae_eap.graph.node import Node, AttentionNode, InputNode, LogitNode, MLPNode
from sae_eap.graph.edge import Edge, attn_edge_types

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


def add_attn_edges(graph: Graph, parent_node: Node, child_attn_node: AttentionNode):
    """Add edges from parent_node to a child node."""
    for edge_type in attn_edge_types():
        edge = Edge(parent_node, child_attn_node, edge_type)
        graph.add_edge(edge)


def add_non_attn_edge(graph: Graph, parent_node: Node, child_node: Node):
    """Add a single edge from parent_node to child_node that is not an attention edge."""
    assert not isinstance(
        child_node, AttentionNode
    ), f"child_node must not be an AttentionNode, but got {child_node}"
    edge = Edge(parent_node, child_node, "na")
    graph.add_edge(edge)


def add_layer_nodes_and_edges(
    graph: Graph,
    layer: int,
    prev_layers_nodes: list[Node],
    parallel_attn_mlp: bool = False,
) -> list[Node]:
    """Add edges for a single layer."""

    attn_nodes = [AttentionNode(layer, head) for head in range(graph.cfg.n_heads)]
    mlp_node = MLPNode(layer)
    # Add the nodes
    for node in attn_nodes:
        graph.add_node(node)
    graph.add_node(mlp_node)

    for node in prev_layers_nodes:
        for attn_node in attn_nodes:
            add_attn_edges(graph, node, attn_node)
        add_non_attn_edge(graph, node, mlp_node)

    if not parallel_attn_mlp:
        # The attention nodes are connected to the MLP node
        for attn_node in attn_nodes:
            add_non_attn_edge(graph, attn_node, mlp_node)
    else:
        # The attention nodes are not connected to the MLP node
        # So we do nothing here
        pass

    # Update prev_layers_nodes
    prev_layers_nodes += attn_nodes
    prev_layers_nodes.append(mlp_node)

    return prev_layers_nodes


def build_graph(
    model_or_config: HookedTransformer | HookedTransformerConfig | dict,
) -> Graph:
    """Build a graph from a model or config."""
    cfg = parse_model_or_config(model_or_config)
    graph = Graph(cfg)

    input_node = InputNode()
    graph.add_node(input_node)
    residual_stream: list[Node] = [input_node]

    for layer in range(graph.cfg.n_layers):
        residual_stream = add_layer_nodes_and_edges(graph, layer, residual_stream)

    logit_node = LogitNode(graph.cfg.n_layers)
    for node in residual_stream:
        add_non_attn_edge(graph, node, logit_node)

    graph.add_node(logit_node)
    return graph
