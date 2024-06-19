from sae_eap.graph.build import build_graph
from sae_eap.graph.index import TensorGraphIndexer


def test_graph_index_does_not_repeat(ts_model):
    graph = build_graph(ts_model)
    graph_indexer = TensorGraphIndexer(graph)

    src_indices_seen = set()
    dest_indices_seen = set()

    for node in graph.src_nodes:
        src_index = graph_indexer.get_src_index(node)
        assert src_index not in src_indices_seen
        src_indices_seen.add(src_index)

    for node in graph.dest_nodes:
        dest_index = graph_indexer.get_dest_index(node)
        assert dest_index not in dest_indices_seen
        dest_indices_seen.add(dest_index)
