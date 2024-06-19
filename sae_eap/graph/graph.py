from __future__ import annotations

from typing import Any, Sequence

import networkx as nx
from transformer_lens import HookedTransformerConfig

from sae_eap.core.constants import GPT_2_SMALL_MODEL_CONFIG
from sae_eap.graph.edge import Edge
from sae_eap.graph.node import Node


class Graph:
    """
    A class to represent a computational graph, which is a DAG.

    Implemented as a wrapper around a networkx.MultiDiGraph.
    """

    cfg: HookedTransformerConfig
    graph: nx.MultiDiGraph

    def __init__(
        self,
        cfg: HookedTransformerConfig = GPT_2_SMALL_MODEL_CONFIG,
        graph: nx.MultiDiGraph | None = None,
    ):
        self.cfg = cfg
        if graph is not None:
            self.graph = graph
        else:
            self.graph = nx.MultiDiGraph()

    """ Basic getter, setter, and utility functions """

    @property
    def nodes(self) -> Sequence[Node]:
        return list(self.graph.nodes)

    @property
    def edges(self) -> Sequence[Edge]:
        return list(self.graph.edges)

    def get_children(self, node: Node) -> set[Node]:
        """Get the children of a node."""
        return set(self.graph.successors(node))

    def get_parents(self, node: Node) -> set[Node]:
        """Get the parents of a node."""
        return set(self.graph.predecessors(node))

    def to_json(self) -> dict[str, Any]:
        """Convert the graph to a JSON object."""
        cfg_data = self.cfg.to_dict()
        graph_data = nx.node_link_data(self.graph)  # type: ignore
        return {"cfg": cfg_data, "graph": graph_data}

    @staticmethod
    def from_json(data: dict[str, Any]) -> Graph:
        """Create a graph from a JSON object."""
        cfg = HookedTransformerConfig.from_dict(data["cfg"])
        graph = nx.node_link_graph(data["graph"])
        return Graph(cfg=cfg, graph=graph)

    def copy(self) -> Graph:
        """Return a copy of the graph."""
        return Graph(cfg=self.cfg, graph=self.graph.copy())  # type: ignore

    """ Methods to manipulate the graph """

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.graph.add_node(node)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.graph.add_edge(
            u_for_edge=edge.parent,
            v_for_edge=edge.child,
            key=edge.type,
        )

    def remove_node(self, node: Node) -> None:
        """Remove a node from the graph."""
        self.graph.remove_node(node)

    def remove_edge(self, edge: Edge) -> None:
        """Remove an edge from the graph."""
        self.graph.remove_edge(edge.parent, edge.child, key=edge.type)

    def prune_dead_ends(self) -> None:
        """Prune dead-end nodes."""
        raise NotImplementedError

    """ Methods to manipulate metadata """

    def get_node_info(self, node: Node) -> dict[str, Any]:
        """Get the node info."""
        return self.graph.nodes[node]

    def set_node_info(self, node: Node, info: dict[str, Any]) -> None:
        """Set the node info."""
        self.graph.nodes[node].update(info)

    def get_edge_info(self, edge: Edge) -> dict[str, Any]:
        """Get the edge info."""
        return self.graph.get_edge_data(edge.parent, edge.child, key=edge.type)  # type: ignore

    def set_edge_info(self, edge: Edge, info: dict[str, Any]) -> None:
        """Set the edge info."""
        self.graph[edge.parent][edge.child][edge.type].update(info)

    """ Syntactic sugar """

    def get_edge_score(self, edge) -> float:
        """Get the edge score."""
        return self.get_edge_info(edge)["score"]

    def get_all_edge_scores(self) -> Sequence[float]:
        """Get the scores of all edges."""
        return [self.get_edge_score(edge) for edge in self.edges]
