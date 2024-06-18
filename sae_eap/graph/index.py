from sae_eap.graph import Graph, Node

Index = tuple[int, ...]


def get_indices(start, end, step: int = 1) -> Index:
    return tuple(range(start, end, step))


class GraphIndexer:
    """A class to index nodes in a graph."""

    node_to_act_index: dict[Node, Index]
    node_to_grad_index: dict[Node, Index]

    def __init__(self, graph: Graph):
        self.reset_index(graph)

    def reset_index(self, graph: Graph):
        node_to_act_index = {}
        curr_act_index = 0
        for node in graph.nodes:
            n_node_outputs = len(node.get_out_hooks())
            node_to_act_index[node] = get_indices(
                curr_act_index, curr_act_index + n_node_outputs
            )
            curr_act_index += n_node_outputs
        self._n_act_index = curr_act_index
        self.node_to_act_index = node_to_act_index

        node_to_grad_index = {}
        curr_n_grad_index = 0
        for node in graph.nodes:
            n_node_inputs = len(node.get_in_hooks())
            node_to_grad_index[node] = get_indices(
                curr_n_grad_index, curr_n_grad_index + n_node_inputs
            )
            curr_n_grad_index += n_node_inputs
        self._n_grad_index = curr_n_grad_index
        self.node_to_grad_index = node_to_grad_index

    @property
    def n_act_index(self):
        return self._n_act_index

    @property
    def n_grad_index(self):
        return self._n_grad_index

    def get_act_index(self, node: Node) -> Index:
        return self.node_to_act_index[node]

    def get_grad_index(self, node: Node) -> Index:
        return self.node_to_grad_index[node]
