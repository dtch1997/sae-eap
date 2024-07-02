import pytest

from sae_eap.graph import Node, Edge, Graph
from sae_eap.prune import sort_edges_by_score_descending, TopNEdgePruner


@pytest.fixture()
def graph():
    nodes = [
        Node(name="A"),
        Node(name="B"),
        Node(name="C"),
    ]

    edges = [
        Edge(nodes[0], nodes[1]),
        Edge(nodes[1], nodes[2]),
    ]

    graph = Graph()
    for node in nodes:
        graph.add_node(node)
    for edge in edges:
        graph.add_edge(edge)

    return graph


def test_sort_edges_by_score_descending(graph):
    edges = list(graph.edges)

    scores = {edges[i].name: float(i) for i in range(len(edges))}

    sorted_edges = sort_edges_by_score_descending(edges, scores)  # type: ignore
    assert sorted_edges == edges[::-1]


def test_top_n_edge_pruner(graph):
    pruner = TopNEdgePruner(1)

    edges = list(graph.edges)
    scores = {edges[i].name: float(i) for i in range(len(edges))}

    pruner.prune(graph, scores)

    assert len(graph.edges) == 1
    assert list(graph.edges)[0].src == Node(name="B")
    assert list(graph.edges)[0].dest == Node(name="C")
