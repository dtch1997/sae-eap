import pytest

from sae_eap.graph.graph import Graph
from sae_eap.graph.node import Node
from sae_eap.graph.edge import Edge


@pytest.fixture
def nodes():
    return [Node(name="A"), Node(name="B"), Node(name="C")]


@pytest.fixture
def edges():
    return [Edge(Node(name="A"), Node(name="B")), Edge(Node(name="B"), Node(name="C"))]


def test_graph_nodes(nodes, edges):
    graph = Graph()
    assert len(graph.nodes) == 0

    for node in nodes:
        graph.add_node(node)

    assert len(graph.nodes) == 3
    assert all(node in graph.nodes for node in nodes)


def test_graph_edges(nodes, edges):
    graph = Graph()
    assert len(graph.edges) == 0

    for edge in edges:
        graph.add_edge(edge)

    assert len(graph.edges) == 2
    assert all(edge in graph.edges for edge in edges)


def test_get_set_node_info(nodes, edges):
    graph = Graph()
    for node in nodes:
        graph.add_node(node)
        info = graph.get_node_info(node)
        assert isinstance(info, dict)
        assert len(info) == 0

        expected_info = {"key": "value"}
        graph.set_node_info(node, expected_info)
        info = graph.get_node_info(node)
        assert info == expected_info


def test_get_set_edge_info(nodes, edges):
    graph = Graph()
    for edge in edges:
        graph.add_edge(edge)
        info = graph.get_edge_info(edge)
        assert isinstance(info, dict)
        assert len(info) == 0

        expected_info = {"key": "value"}
        graph.set_edge_info(edge, expected_info)
        info = graph.get_edge_info(edge)
        assert info == expected_info


# def test_get_edge_score():
#     graph = Graph()
#     node1 = Node("A")
#     node2 = Node("B")
#     node3 = Node("C")
#     edge1 = Edge(node1, node2, "edge1")
#     edge2 = Edge(node2, node3, "edge2")
#     graph.add_edge(edge1)
#     graph.add_edge(edge2)

#     edge_score = graph.get_edge_score(edge1)
#     assert isinstance(edge_score, float)


# def test_get_all_edge_scores():
#     graph = Graph()
#     node1 = Node("A")
#     node2 = Node("B")
#     node3 = Node("C")
#     edge1 = Edge(node1, node2, "edge1")
#     edge2 = Edge(node2, node3, "edge2")
#     graph.add_edge(edge1)
#     graph.add_edge(edge2)

#     all_edge_scores = graph.get_all_edge_scores()
#     assert isinstance(all_edge_scores, list)
#     assert all(isinstance(score, float) for score in all_edge_scores)
