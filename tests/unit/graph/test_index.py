from sae_eap.graph.build import build_graph
from sae_eap.graph.index import GraphIndexer


def get_n_act_index_of_model_only_graph(cfg):
    """Get the number of activation indices in a graph without any SAEs.

    Formula:
    - InputNode: 1
    - Attention Node: n_head x n_layers
    - MLP Node: n_layers
    - Logit Node: 0
    """
    return 1 + cfg.n_layers * (cfg.n_heads + 1)


def get_n_grad_index_of_model_only_graph(cfg):
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
    assert graph_indexer.n_act_index == get_n_act_index_of_model_only_graph(graph.cfg)
    assert graph_indexer.n_grad_index == get_n_grad_index_of_model_only_graph(graph.cfg)

    for node in graph.nodes:
        act_index = graph_indexer.get_act_index(node)
        grad_index = graph_indexer.get_grad_index(node)
        assert len(act_index) == len(node.get_out_hooks())
        assert len(grad_index) == len(node.get_in_hooks())


def test_graph_index_does_not_repeat(ts_model):
    graph = build_graph(ts_model)
    graph_indexer = GraphIndexer(graph)

    act_indices_seen = set()
    grad_indices_seen = set()

    for node in graph.nodes:
        act_index = graph_indexer.get_act_index(node)
        grad_index = graph_indexer.get_grad_index(node)

        for a in act_index:
            assert a not in act_indices_seen
            act_indices_seen.add(act_index)

        for g in grad_index:
            assert g not in grad_indices_seen
            grad_indices_seen.add(grad_index)
