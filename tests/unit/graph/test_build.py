from sae_eap.graph.build import build_graph
from tests.unit.helpers import load_model_cached, SOLU_1L_MODEL


def test_build_graph_has_correct_num_nodes_and_edges():
    model = load_model_cached(SOLU_1L_MODEL)
    graph = build_graph(model)
    assert len(graph.nodes) == 11
    assert len(graph.edges) == 43
