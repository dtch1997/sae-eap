from sae_eap.graph.build import build_graph
from sae_eap.graph.index import GraphIndexer


def get_n_outputs_of_model_only_graph(cfg):
    """Get the number of activation indices in a graph without any SAEs.

    Formula:
    - InputNode: 1
    - Attention Node: n_head x n_layers
    - MLP Node: n_layers
    - Logit Node: 0
    """
    return 1 + cfg.n_layers * (cfg.n_heads + 1)


def get_n_inputs_of_model_only_graph(cfg):
    """Get the number of gradient indices in a graph without any SAEs.

    Formula:
    - InputNode: 0
    - Attention Node: 3 x n_head x n_layers
    - MLP Node: n_layers
    - Logit Node: 1
    """
    return cfg.n_layers * (3 * cfg.n_heads + 1) + 1


def test_graph_indexer_basic(ts_model):
    graph = build_graph(ts_model)
    graph_indexer = GraphIndexer(graph)
    assert graph_indexer.n_outputs == get_n_outputs_of_model_only_graph(graph.cfg)
    assert graph_indexer.n_inputs == get_n_inputs_of_model_only_graph(graph.cfg)

    for node in graph.nodes:
        output_index = graph_indexer.get_output_index(node)
        input_index = graph_indexer.get_input_index(node)
        assert len(output_index) == len(node.get_out_hooks())
        assert len(input_index) == len(node.get_in_hooks())


def test_graph_index_does_not_repeat(ts_model):
    graph = build_graph(ts_model)
    graph_indexer = GraphIndexer(graph)

    output_indices_seen = set()
    input_indices_seen = set()

    for node in graph.nodes:
        for o in graph_indexer.get_output_index(node):
            assert o not in output_indices_seen
            output_indices_seen.add(o)

        for g in graph_indexer.get_input_index(node):
            assert g not in input_indices_seen
            input_indices_seen.add(g)
