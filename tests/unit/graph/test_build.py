from sae_eap.graph.build import build_graph
from sae_eap.graph.graph import TensorGraph
from tests.unit.helpers import load_model_cached, SOLU_1L_MODEL


def test_build_graph_can_run():
    model = load_model_cached(SOLU_1L_MODEL)
    graph = build_graph(model)
    assert isinstance(graph, TensorGraph)
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0


def test_build_graph_has_correct_num_nodes():
    model = load_model_cached(SOLU_1L_MODEL)
    graph = build_graph(model)

    n_layers = 1
    n_heads = 8

    n_input_nodes = 1
    n_output_nodes = 1
    n_attn_nodes = n_layers * n_heads * (3 + 1)
    n_mlp_nodes = n_layers * 2
    n_nodes = n_input_nodes + n_output_nodes + n_attn_nodes + n_mlp_nodes

    assert len(graph.nodes) == n_nodes


def test_build_graph_has_correct_num_edges():
    model = load_model_cached(SOLU_1L_MODEL)
    graph = build_graph(model)

    n_heads = 8

    n_edges_from_input = 3 * n_heads + 1 + 1
    n_edges_from_attn = n_heads * (1 + 1)  # 1 MLP, 1 Output
    n_edges_from_mlp = 1
    n_edges = n_edges_from_input + n_edges_from_attn + n_edges_from_mlp

    assert len(graph.edges) == n_edges
