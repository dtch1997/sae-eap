from typing import Any, Sequence
from sae_eap.graph.node import Node
from sae_eap.graph.edge import Edge
from sae_eap.core.constants import GPT_2_SMALL_MODEL_CONFIG
from transformer_lens import HookedTransformerConfig

import networkx as nx


class Graph:
    """
    A class to represent a computational graph, which is a DAG.

    Implemented as a wrapper around a networkx.DiGraph.
    """

    cfg: HookedTransformerConfig
    graph: nx.DiGraph

    """ Main APIs """

    @property
    def nodes(self) -> Sequence[Node]:
        """Dictionary of (node_name, node) pairs."""
        raise NotImplementedError

    @property
    def edges(self) -> Sequence[Edge]:
        """Dictionary of (edge_name, edge) pairs."""
        raise NotImplementedError

    @property
    def n_forward_nodes(self) -> int:
        """The number of forward nodes in the graph."""
        raise NotImplementedError

    @property
    def n_backward_nodes(self) -> int:
        """The number of backward nodes in the graph."""
        raise NotImplementedError

    def get_children(self, node: Node) -> set[Node]:
        """Get the children of a node."""
        raise NotImplementedError

    def get_parents(self, node: Node) -> set[Node]:
        """Get the parents of a node."""
        raise NotImplementedError

    def to_json(self) -> dict[str, Any]:
        """Convert the graph to a JSON object."""
        return nx.node_link_data(self.graph)  # type: ignore

    @staticmethod
    def from_json(data: dict[str, Any]) -> "Graph":
        """Create a graph from a JSON object."""
        return Graph(nx.node_link_graph(data))

    """ Helper functions """

    def __init__(
        self,
        cfg: HookedTransformerConfig = GPT_2_SMALL_MODEL_CONFIG,
        graph: nx.DiGraph | None = None,
    ):
        self.cfg = cfg
        if graph is not None:
            self.graph = graph
        else:
            self.graph = nx.DiGraph()

    def copy(self) -> "Graph":
        """Return a copy of the graph."""
        return Graph(self.graph.copy())  # type: ignore

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        raise NotImplementedError

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        # NOTE: This should do a few things:
        # 1. Add the edge to the graph.
        # 2. Add the child to the parent's children.
        # 3. Add the parent to the child's parents.
        raise NotImplementedError

    def remove_node(self, node: Node) -> None:
        """Remove a node from the graph."""
        # NOTE: This should do a few things:
        # 1. Remove the node from the graph.
        # 2. Remove all edges connected to the node.
        # 3. Remove the node from its parents' children.
        # 4. Remove the node from its children's parents.
        raise NotImplementedError

    def remove_edge(self, edge: Edge) -> None:
        """Remove an edge from the graph."""
        # NOTE: This should do a few things:
        # 1. Remove the edge from the graph.
        # 2. Remove the child from the parent's children.
        # 3. Remove the parent from the child's parents.
        raise NotImplementedError

    def prune_dead_ends(self) -> None:
        """Prune dead-end nodes."""
        raise NotImplementedError

    def get_edge_info(self, edge) -> dict[Any, Any]:
        """Get the edge info."""
        raise NotImplementedError

    """ Syntactic sugar functions """

    def get_edge_score(self, edge) -> float:
        """Get the edge score."""
        return self.get_edge_info(edge)["score"]

    def get_all_edge_scores(self) -> Sequence[float]:
        """Get the scores of all edges."""
        return [self.get_edge_score(edge) for edge in self.edges]
