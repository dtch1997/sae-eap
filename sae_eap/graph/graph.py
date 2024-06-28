from __future__ import annotations

from typing import Any, Sequence, TypeVar, Generic, Type, cast

import networkx as nx
from transformer_lens import HookedTransformerConfig

from sae_eap.core.constants import GPT_2_SMALL_MODEL_CONFIG
from sae_eap.graph.edge import Edge, TensorEdge
from sae_eap.graph.node import Node, TensorNode

TNode = TypeVar("TNode", bound=Node)
TEdge = TypeVar("TEdge", bound=Edge)


class Graph(Generic[TNode, TEdge]):
    """
    A class to represent a graph, which is a collection of nodes and edges.

    Implemented as a wrapper around a NetworkX DiGraph.
    """

    graph: nx.DiGraph
    node_cls: Type[TNode]
    edge_cls: Type[TEdge]

    def __init__(
        self,
        graph: nx.DiGraph | None = None,
        node_cls: Type[TNode] = Node,
        edge_cls: Type[TEdge] = Edge,
    ):
        if graph is not None:
            self.graph = graph
        else:
            self.graph = nx.DiGraph()

        self.node_cls = node_cls
        self.edge_cls = edge_cls

    """ Basic getter, setter, and utility functions """

    @property
    def nodes(self) -> Sequence[TNode]:
        return list(self.graph.nodes)

    @property
    def src_nodes(self) -> Sequence[TNode]:
        """Get the source nodes in the graph."""
        return [node for node in self.graph.nodes if node.is_src]  # type: ignore

    @property
    def dest_nodes(self) -> Sequence[TNode]:
        """Get the destination nodes in the graph."""
        return [node for node in self.graph.nodes if node.is_dest]  # type: ignore

    @property
    def edges(self) -> Sequence[TEdge]:
        edges = []
        for src, dest in self.graph.edges:
            edges.append(self.edge_cls(src, dest))
        return edges

    def has_node(self, node: TNode) -> bool:
        """Check if a node exists in the graph."""
        return self.graph.has_node(node)

    def has_edge(self, edge: TEdge) -> bool:
        """Check if an edge exists in the graph."""
        return self.graph.has_edge(edge.parent, edge.child)

    def get_children(self, node: TNode) -> set[TNode]:
        """Get the children of a node."""
        return set(self.graph.successors(node))

    def get_parents(self, node: TNode) -> set[TNode]:
        """Get the parents of a node."""
        return set(self.graph.predecessors(node))

    def to_json(self) -> dict[str, Any]:
        """Convert the graph to a JSON object."""
        graph_data = nx.node_link_data(self.graph)
        return {"graph": graph_data}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Graph:
        """Create a graph from a JSON object."""
        graph = nx.node_link_graph(data["graph"])
        return cls(graph=graph)

    def copy(self) -> Graph:
        """Return a copy of the graph."""
        new_nx_digraph = cast(nx.DiGraph, self.graph.copy())
        return Graph(graph=new_nx_digraph)

    """ Methods to manipulate the graph """

    def add_node(self, node: TNode) -> None:
        """Add a node to the graph."""
        self.graph.add_node(node)

    def add_edge(self, edge: TEdge) -> None:
        """Add an edge to the graph."""
        self.graph.add_edge(
            u_of_edge=edge.parent,
            v_of_edge=edge.child,
        )

    def remove_node(self, node: TNode) -> None:
        """Remove a node from the graph."""
        self.graph.remove_node(node)

    def remove_edge(self, edge: TEdge) -> None:
        """Remove an edge from the graph."""
        self.graph.remove_edge(edge.parent, edge.child)

    """ Methods to manipulate metadata """

    def get_node_info(self, node: TNode) -> dict[str, Any]:
        """Get the node info."""
        return self.graph.nodes[node]

    def set_node_info(self, node: TNode, info: dict[str, Any]) -> None:
        """Set the node info."""
        self.graph.nodes[node].update(info)

    def get_edge_info(self, edge: TEdge) -> dict[str, Any]:
        """Get the edge info."""
        return self.graph.get_edge_data(edge.parent, edge.child)  # type: ignore

    def set_edge_info(self, edge: TEdge, info: dict[str, Any]) -> None:
        """Set the edge info."""
        self.graph[edge.parent][edge.child].update(info)

    """ Syntactic sugar """

    def get_edge_score(self, edge: TEdge) -> float:
        """Get the edge score."""
        return self.get_edge_info(edge)["score"]

    def get_all_edge_scores(self) -> Sequence[float]:
        """Get the scores of all edges."""
        return [self.get_edge_score(edge) for edge in self.edges]


class TensorGraph(Graph[TensorNode, TensorEdge]):
    """Represents the computational graph of a transformer model."""

    cfg: HookedTransformerConfig

    def __init__(
        self,
        cfg: HookedTransformerConfig = GPT_2_SMALL_MODEL_CONFIG,
        graph: nx.DiGraph | None = None,
    ):
        super().__init__(graph=graph, node_cls=TensorNode, edge_cls=TensorEdge)
        self.cfg = cfg

    def to_json(self) -> dict[str, Any]:
        """Convert the graph to a JSON object."""
        cfg_data = self.cfg.to_dict()
        graph_data = nx.node_link_data(self.graph)  # type: ignore
        return {"cfg": cfg_data, "graph": graph_data}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TensorGraph:
        """Create a graph from a JSON object."""
        cfg = HookedTransformerConfig.from_dict(data["cfg"])
        graph = nx.node_link_graph(data["graph"])
        return cls(cfg=cfg, graph=graph)

    def copy(self) -> TensorGraph:
        """Return a copy of the graph."""
        new_nx_digraph = cast(nx.DiGraph, self.graph.copy())
        return TensorGraph(cfg=self.cfg, graph=new_nx_digraph)
