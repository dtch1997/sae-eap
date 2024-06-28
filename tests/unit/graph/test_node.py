from sae_eap.graph import Node


def test_node_is_hashable():
    node = Node(name="A")
    hash(node)
