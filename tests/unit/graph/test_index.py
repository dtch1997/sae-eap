import pytest
from sae_eap.graph.build import build_graph


@pytest.mark.xfail
def test_graph_indexer(ts_model):
    graph = build_graph(ts_model)
    assert graph.n_forward == 1 + graph.cfg["n_layers"] * (graph.cfg["n_heads"] + 1)
    assert (
        graph.n_backward == graph.cfg["n_layers"] * (3 * graph.cfg["n_heads"] + 1) + 1
    )
