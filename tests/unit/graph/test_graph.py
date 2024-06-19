import pytest

from sae_eap.graph.graph import Graph
from sae_eap.graph.node import InputNode, MLPNode, AttentionNode, LogitNode


@pytest.fixture
def nodes():
    return [
        InputNode(),
        AttentionNode(0, 0),
        MLPNode(0),
        LogitNode(1),
    ]


def test_graph_nodes(nodes):
    graph = Graph()
    assert len(graph.nodes) == 0

    for node in nodes:
        graph.add_node(node)

    assert len(graph.nodes) == 4
    assert all(node in graph.nodes for node in nodes)


# def test_graph_edges():
#     graph = Graph()
#     node1 = Node("A")
#     node2 = Node("B")
#     node3 = Node("C")
#     edge1 = Edge(node1, node2, "edge1")
#     edge2 = Edge(node2, node3, "edge2")
#     graph.add_edge(edge1)
#     graph.add_edge(edge2)

#     assert len(graph.edges) == 2
#     assert edge1 in graph.edges
#     assert edge2 in graph.edges


# def test_get_children():
#     graph = Graph()
#     node1 = Node("A")
#     node2 = Node("B")
#     node3 = Node("C")
#     edge1 = Edge(node1, node2, "edge1")
#     edge2 = Edge(node2, node3, "edge2")
#     graph.add_edge(edge1)
#     graph.add_edge(edge2)

#     assert graph.get_children(node1) == {node2}
#     assert graph.get_children(node2) == {node3}
#     assert graph.get_children(node3) == set()


# def test_get_parents():
#     graph = Graph()
#     node1 = Node("A")
#     node2 = Node("B")
#     node3 = Node("C")
#     edge1 = Edge(node1, node2, "edge1")
#     edge2 = Edge(node2, node3, "edge2")
#     graph.add_edge(edge1)
#     graph.add_edge(edge2)

#     assert graph.get_parents(node1) == set()
#     assert graph.get_parents(node2) == {node1}
#     assert graph.get_parents(node3) == {node2}


# def test_to_json():
#     graph = Graph()
#     node1 = Node("A")
#     node2 = Node("B")
#     node3 = Node("C")
#     edge1 = Edge(node1, node2, "edge1")
#     edge2 = Edge(node2, node3, "edge2")
#     graph.add_edge(edge1)
#     graph.add_edge(edge2)

#     json_data = graph.to_json()
#     assert isinstance(json_data, dict)
#     assert "cfg" in json_data
#     assert "graph" in json_data


# def test_from_json():
#     graph = Graph()
#     node1 = Node("A")
#     node2 = Node("B")
#     node3 = Node("C")
#     edge1 = Edge(node1, node2, "edge1")
#     edge2 = Edge(node2, node3, "edge2")
#     graph.add_edge(edge1)
#     graph.add_edge(edge2)

#     json_data = graph.to_json()
#     new_graph = Graph.from_json(json_data)

#     assert isinstance(new_graph, Graph)
#     assert len(new_graph.nodes) == len(graph.nodes)
#     assert len(new_graph.edges) == len(graph.edges)
#     assert all(node in new_graph.nodes for node in graph.nodes)
#     assert all(edge in new_graph.edges for edge in graph.edges)


# def test_copy():
#     graph = Graph()
#     node1 = Node("A")
#     node2 = Node("B")
#     node3 = Node("C")
#     edge1 = Edge(node1, node2, "edge1")
#     edge2 = Edge(node2, node3, "edge2")
#     graph.add_edge(edge1)
#     graph.add_edge(edge2)

#     new_graph = graph.copy()

#     assert isinstance(new_graph, Graph)
#     assert len(new_graph.nodes) == len(graph.nodes)
#     assert len(new_graph.edges) == len(graph.edges)
#     assert all(node in new_graph.nodes for node in graph.nodes)
#     assert all(edge in new_graph.edges for edge in graph.edges)
#     assert new_graph is not graph


# def test_remove_node():
#     graph = Graph()
#     node1 = Node("A")
#     node2 = Node("B")
#     node3 = Node("C")
#     edge1 = Edge(node1, node2, "edge1")
#     edge2 = Edge(node2, node3, "edge2")
#     graph.add_edge(edge1)
#     graph.add_edge(edge2)

#     graph.remove_node(node2)

#     assert len(graph.nodes) == 2
#     assert node2 not in graph.nodes
#     assert edge1 not in graph.edges
#     assert edge2 not in graph.edges


# def test_remove_edge():
#     graph = Graph()
#     node1 = Node("A")
#     node2 = Node("B")
#     node3 = Node("C")
#     edge1 = Edge(node1, node2, "edge1")
#     edge2 = Edge(node2, node3, "edge2")
#     graph.add_edge(edge1)
#     graph.add_edge(edge2)

#     graph.remove_edge(edge1)

#     assert len(graph.edges) == 1
#     assert edge1 not in graph.edges
#     assert edge2 in graph.edges


# def test_get_node_info():
#     graph = Graph()
#     node1 = Node("A")
#     node2 = Node("B")
#     node3 = Node("C")
#     edge1 = Edge(node1, node2, "edge1")
#     edge2 = Edge(node2, node3, "edge2")
#     graph.add_edge(edge1)
#     graph.add_edge(edge2)

#     node_info = graph.get_node_info(node1)
#     assert isinstance(node_info, dict)


# def test_set_node_info():
#     graph = Graph()
#     node1 = Node("A")
#     node2 = Node("B")
#     node3 = Node("C")
#     edge1 = Edge(node1, node2, "edge1")
#     edge2 = Edge(node2, node3, "edge2")
#     graph.add_edge(edge1)
#     graph.add_edge(edge2)

#     info = {"key": "value"}
#     graph.set_node_info(node1, info)

#     node_info = graph.get_node_info(node1)
#     assert node_info == info


# def test_get_edge_info():
#     graph = Graph()
#     node1 = Node("A")
#     node2 = Node("B")
#     node3 = Node("C")
#     edge1 = Edge(node1, node2, "edge1")
#     edge2 = Edge(node2, node3, "edge2")
#     graph.add_edge(edge1)
#     graph.add_edge(edge2)

#     edge_info = graph.get_edge_info(edge1)
#     assert isinstance(edge_info, dict)


# def test_set_edge_info():
#     graph = Graph()
#     node1 = Node("A")
#     node2 = Node("B")
#     node3 = Node("C")
#     edge1 = Edge(node1, node2, "edge1")
#     edge2 = Edge(node2, node3, "edge2")
#     graph.add_edge(edge1)
#     graph.add_edge(edge2)

#     info = {"key": "value"}
#     graph.set_edge_info(edge1, info)

#     edge_info = graph.get_edge_info(edge1)
#     assert edge_info == info


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
