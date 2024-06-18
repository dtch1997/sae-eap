from sae_eap.graph import Graph, Node


class GraphIndexer:
    """A class to index nodes in a graph."""

    def __init__(self, graph: Graph):
        node_to_act_index = {}
        node_to_grad_index = {}
        for i, node in enumerate(graph.nodes):
            node_to_act_index[node] = i
            for node_input in enumerate(node.inputs):
                node_to_grad_index[node_input] = len(node_to_grad_index)

    @property
    def n_act_index(self):
        raise NotImplementedError

    @property
    def n_grad_index(self):
        raise NotImplementedError

    def get_act_index(self, src_node: Node):
        """Get the forward index of a node in the graph."""
        raise NotImplementedError

    def get_grad_index(graph: Graph, dest_node: Node):
        """Get the backward index of a node in the graph."""
        raise NotImplementedError
