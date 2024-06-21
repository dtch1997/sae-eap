import pytest

from sae_eap.core.types import Model
from sae_eap.attribute import (
    make_cache_hooks_and_tensors,
    get_model_caches,
    compute_node_act_cache,
    compute_node_grad_cache,
)
from sae_eap.graph.build import build_graph
from sae_eap.graph.index import TensorGraphIndexer

from tests.integration.attribute.helpers import make_single_prompt_handler


def test_hook_lengths(ts_model: Model):
    graph = build_graph(ts_model.cfg)
    handler = make_single_prompt_handler(ts_model)
    hooks, _ = make_cache_hooks_and_tensors(
        graph, handler.get_batch_size(), handler.get_n_pos()
    )

    # Test hook lengths
    assert len(hooks.fwd_hooks_clean) == len(graph.src_nodes)
    assert len(hooks.fwd_hooks_corrupt) == len(graph.src_nodes)
    assert len(hooks.bwd_hooks_clean) == len(graph.dest_nodes)


@pytest.mark.skip("Not implemented")
def test_forward_hooks_work(ts_model: Model):
    """Test that running model with forward hooks modifies the act cache."""
    raise NotImplementedError


@pytest.mark.skip("Not implemented")
def test_backward_hooks_work(ts_model: Model):
    """Test that running model with forward hooks modifies the act cache."""
    raise NotImplementedError


def test_model_cache_has_correct_tensor_shape(
    ts_model: Model,
):
    graph = build_graph(ts_model.cfg)
    handler = make_single_prompt_handler(ts_model)
    caches = get_model_caches(ts_model, graph, handler)

    # Test tensor shapes
    for cache in caches:
        assert cache.batch_size == handler.get_batch_size()
        assert cache.n_pos == handler.get_n_pos()
        assert cache.d_model == ts_model.cfg.d_model


@pytest.mark.xfail(reason="Shape error in compute_node_act_cache")
def test_compute_node_caches(ts_model: Model):
    graph = build_graph(ts_model.cfg)
    indexer = TensorGraphIndexer(graph)
    handler = make_single_prompt_handler(ts_model)
    caches = get_model_caches(ts_model, graph, handler)

    # Compute node caches
    act_cache = compute_node_act_cache(indexer.src_index, caches[0])
    grad_cache = compute_node_grad_cache(indexer.dest_index, caches[1])

    # Test tensor shapes
    for cache in [act_cache, grad_cache]:
        batch_size, n_pos, _, d_model = cache.shape
        assert batch_size == handler.get_batch_size()
        assert n_pos == handler.get_n_pos()
        assert d_model == ts_model.cfg.d_model
