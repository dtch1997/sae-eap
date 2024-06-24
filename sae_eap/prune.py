"""Pruning algorithms for the graph."""

from __future__ import annotations

import abc

from sae_eap.graph.edge import TensorEdge
from sae_eap.graph.node import SrcNode, DestNode, AttentionSrcNode
from sae_eap.graph.build import build_attn_nodes, build_mlp_nodes
from sae_eap.graph.graph import TensorGraph
from sae_eap.attribute import AttributionScores


class Pruner(abc.ABC):
    """Base class for pruning algorithms."""

    absolute: bool

    def __init__(self, absolute: bool = True):
        self.absolute = absolute

    def maybe_abs(self, score: float) -> float:
        return abs(score) if self.absolute else score

    @abc.abstractmethod
    def prune(self, graph: TensorGraph, scores: AttributionScores):
        raise NotImplementedError


class PruningPipeline(Pruner):
    """Pipeline for pruning algorithms."""

    def __init__(self, pruners: list[Pruner] = []):
        self.pruners = pruners

    def add_pruner(self, pruner: Pruner) -> PruningPipeline:
        self.pruners.append(pruner)
        return self

    def reset_pruners(self) -> PruningPipeline:
        self.pruners = []
        return self

    def prune(self, graph: TensorGraph, scores: AttributionScores):
        for pruner in self.pruners:
            pruner.prune(graph, scores)
        return graph


class DeadNodePruner(Pruner):
    """Prunes away dead nodes.

    A node is dead if any of the following conditions are met:
    - The node is a source node with no children
    - The node is a destination node with no parents.
    """

    def prune(self, graph: TensorGraph, scores: AttributionScores) -> None:
        for node in graph.src_nodes:
            if len(graph.get_children(node)) == 0:
                graph.remove_node(node)

        for node in graph.dest_nodes:
            if len(graph.get_parents(node)) == 0:
                graph.remove_node(node)


class ThresholdEdgePruner(Pruner):
    """Prunes edges based on a threshold."""

    def __init__(self, threshold: float, absolute: bool = True):
        super().__init__(absolute)
        self.threshold = threshold

    def prune(self, graph: TensorGraph, scores: AttributionScores) -> None:
        # Remove edges with score below threshold
        for edge in graph.edges:
            score = scores[edge.name]
            score = self.maybe_abs(score)
            if score < self.threshold:
                graph.remove_edge(edge)


class TopNEdgePruner(Pruner):
    """Prunes edges based on the top N edges."""

    def __init__(self, n_edges: int, absolute: bool = True):
        super().__init__(absolute)
        self.n_edges = n_edges

    def prune(self, graph: TensorGraph, scores: AttributionScores) -> None:
        edges_sorted_by_score_descending = sorted(
            list(graph.edges),
            key=lambda edge: self.maybe_abs(scores[edge.name]),
            reverse=True,
        )

        for node in graph.nodes:
            graph.remove_node(node)

        for edge in edges_sorted_by_score_descending[: self.n_edges]:
            graph.add_edge(edge)


def get_dest_nodes_for_src_node(graph, src_node: SrcNode) -> list[DestNode]:
    if "mlp" in src_node.name:
        _, dest_node = build_mlp_nodes(graph)
        return [dest_node]

    elif "attn" in src_node.name:
        assert isinstance(src_node, AttentionSrcNode)
        _, dest_nodes = build_attn_nodes(graph, src_node.head_index)
        return dest_nodes  # type: ignore

    else:
        raise ValueError(f"Unknown node type: {src_node.name}")


def get_incoming_edges_for_dest_node(
    graph: TensorGraph, dest_node: DestNode
) -> list[TensorEdge]:
    incoming_edges = []
    for parent in graph.get_parents(dest_node):
        assert isinstance(parent, SrcNode)
        edge = graph.edge_cls(parent, dest_node)
        incoming_edges.append(edge)
    return incoming_edges


class GreedyEdgePruner(Pruner):
    """Prunes edges based on a greedy algorithm."""

    def __init__(self, n_edges: int, absolute: bool = True):
        super().__init__(absolute)
        self.n_edges = n_edges

    def prune(self, graph: TensorGraph, scores: AttributionScores) -> None:
        # TODO: the implementation is a bit cursed
        raise NotImplementedError

        # output_node = build_output_node(graph.cfg.n_layers)
        # candidate_edges = sorted(
        #     get_incoming_edges_for_dest_node(graph, output_node),
        #     key=lambda edge: self.maybe_abs(scores[edge.name]),
        #     reverse=True,
        # )

        # edges = heapq.merge(
        #     candidate_edges, key=lambda edge: self.maybe_abs(scores[edge.name]), reverse=True
        # )

        # for node in graph.nodes:
        #     graph.remove_node(node)

        # n_edges = self.n_edges
        # while n_edges > 0:
        #     n_edges -= 1
        #     top_edge = edges[0]
        #     src = top_edge.src
        #     for dest_node in get_dest_nodes_for_src_node(graph, src):

        #     parent_parent_edges = sorted(
        #         [parent_edge for parent_edge in parent.parent_edges],
        #         key=lambda edge: abs_id(edge.score),
        #         reverse=True,
        #     )
        #     edges = heapq.merge(
        #         edges,
        #         parent_parent_edges,
        #         key=lambda edge: abs_id(edge.score),
        #         reverse=True,
        #     )
