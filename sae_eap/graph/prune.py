"""Pruning algorithms for the graph."""

import abc
from sae_eap.graph.graph import Graph


class Pruner(abc.ABC):
    """Base class for pruning algorithms."""

    @abc.abstractmethod
    def apply(self, graph: Graph) -> Graph:
        raise NotImplementedError


class ThresholdPruner(Pruner):
    """Prunes edges based on a threshold."""

    def __init__(self, threshold: float, absolute: bool = True):
        self.threshold = threshold
        self.absolute = absolute

    def apply(self, graph: Graph) -> Graph:
        new_graph = graph.copy()

        # Remove edges with score below threshold
        for edge in new_graph.edges:
            score = new_graph.get_edge_score(edge)
            if self.absolute:
                score = abs(score)
            if score < self.threshold:
                new_graph.remove_edge(edge)

        return new_graph


class TopNPruner(Pruner):
    """Prunes edges based on the top N edges."""

    def __init__(self, n: int, absolute: bool = True):
        self.n = n
        self.absolute = absolute

    def apply(self, graph: Graph) -> Graph:
        new_graph = graph.copy()
        # TODO: implement
        raise NotImplementedError
        return new_graph

    # def apply_topn(self, n: int, absolute: bool):
    #     a = abs if absolute else lambda x: x
    #     for node in self.nodes.values():
    #         node.in_graph = False

    #     sorted_edges = sorted(
    #         list(self.edges.values()), key=lambda edge: a(edge.score), reverse=True
    #     )
    #     for edge in sorted_edges[:n]:
    #         edge.in_graph = True
    #         edge.parent.in_graph = True
    #         edge.child.in_graph = True

    #     for edge in sorted_edges[n:]:
    #         edge.in_graph = False


class GreedyPruner(Pruner):
    """Prunes edges based on a greedy algorithm."""

    def __init__(self, absolute: bool = True):
        self.absolute = absolute

    def apply(self, graph: Graph) -> Graph:
        new_graph = graph.copy()
        # TODO: Implement
        raise NotImplementedError
        return new_graph

    # def apply_greedy(self, n_edges, reset=True, absolute: bool = True):
    #     if reset:
    #         for node in self.nodes.values():
    #             node.in_graph = False
    #         for edge in self.edges.values():
    #             edge.in_graph = False
    #         self.nodes["logits"].in_graph = True

    #     def abs_id(s: float):
    #         return abs(s) if absolute else s

    #     candidate_edges = sorted(
    #         [edge for edge in self.edges.values() if edge.child.in_graph],
    #         key=lambda edge: abs_id(edge.score),
    #         reverse=True,
    #     )

    #     edges = heapq.merge(
    #         candidate_edges, key=lambda edge: abs_id(edge.score), reverse=True
    #     )
    #     while n_edges > 0:
    #         n_edges -= 1
    #         top_edge = next(edges)
    #         top_edge.in_graph = True
    #         parent = top_edge.parent
    #         if not parent.in_graph:
    #             parent.in_graph = True
    #             parent_parent_edges = sorted(
    #                 [parent_edge for parent_edge in parent.parent_edges],
    #                 key=lambda edge: abs_id(edge.score),
    #                 reverse=True,
    #             )
    #             edges = heapq.merge(
    #                 edges,
    #                 parent_parent_edges,
    #                 key=lambda edge: abs_id(edge.score),
    #                 reverse=True,
    #             )
