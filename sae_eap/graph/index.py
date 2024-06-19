from sae_eap.graph import Graph, Node

Index = tuple[int, ...]


def get_indices(start, end, step: int = 1) -> Index:
    return tuple(range(start, end, step))


class GraphIndexer:
    """A class to index nodes in a graph."""

    node_output_index: dict[Node, Index]
    node_input_index: dict[Node, Index]

    def __init__(self, graph: Graph):
        self.build_index(graph)

    def build_index(self, graph: Graph):
        self._build_output_index(graph)
        self._build_input_index(graph)

    def _build_output_index(self, graph: Graph):
        node_output_index = {}
        n_outputs = 0
        for node in graph.nodes:
            n_node_outputs = len(node.get_out_hooks())
            node_output_index[node] = get_indices(n_outputs, n_outputs + n_node_outputs)
            n_outputs += n_node_outputs
        self._n_outputs = n_outputs
        self.node_output_index = node_output_index

    def _build_input_index(self, graph: Graph):
        node_input_index = {}
        n_inputs = 0
        for node in graph.nodes:
            n_node_inputs = len(node.get_in_hooks())
            node_input_index[node] = get_indices(n_inputs, n_inputs + n_node_inputs)
            n_inputs += n_node_inputs
        self._n_input = n_inputs
        self.node_input_index = node_input_index

    @property
    def n_outputs(self):
        return self._n_outputs

    @property
    def n_inputs(self):
        return self._n_input

    def get_output_index(self, node: Node) -> Index:
        return self.node_output_index[node]

    def get_input_index(self, node: Node) -> Index:
        return self.node_input_index[node]

    # def get_edge_index(self, edge: Edge) -> tuple[Index, Index]:
    #     """ Get the output and input indices for an edge."""
    #     src_node = edge.src_node
    #     src_output_index = self.get_output_index(src_node)
    #     dest_node = edge.dest_node
    #     dest_input_index = self.get_input_index(dest_node)

    #     # NOTE: The edge index will only go to one of the dest_node's input hooks...
