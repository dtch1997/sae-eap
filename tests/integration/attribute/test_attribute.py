import pytest

from sae_eap.core.types import Model
from sae_eap.attribute import (
    make_cache_hooks_and_dicts,
    compute_model_caches,
    compute_node_act_cache,
    compute_node_grad_cache,
)
from sae_eap.graph.build import build_graph
from sae_eap.graph.index import TensorGraphIndexer

from tests.integration.attribute.helpers import make_single_prompt_handler


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
    hooks, caches = make_cache_hooks_and_dicts(graph)
    caches = compute_model_caches(ts_model, hooks, caches, handler)

    # Test tensor shapes
    for cache in caches:
        assert cache.batch_size == handler.get_batch_size()
        assert cache.n_pos == handler.get_n_pos()
        assert cache.d_model == ts_model.cfg.d_model


def test_node_act_has_correct_tensor_shape(ts_model: Model):
    graph = build_graph(ts_model.cfg)
    handler = make_single_prompt_handler(ts_model)
    hooks, caches = make_cache_hooks_and_dicts(graph)
    caches = compute_model_caches(ts_model, hooks, caches, handler)
    act_cache = caches.act_cache

    # Test tensor shapes
    for src_node in graph.src_nodes:
        hook_act = act_cache[src_node.hook]
        assert len(hook_act.shape) >= 3
        act = src_node.get_act(hook_act)
        assert len(act.shape) == 3


def test_node_cache_has_correct_tensor_shape(ts_model: Model):
    graph = build_graph(ts_model.cfg)
    indexer = TensorGraphIndexer(graph)
    handler = make_single_prompt_handler(ts_model)
    hooks, caches = make_cache_hooks_and_dicts(graph)
    caches = compute_model_caches(ts_model, hooks, caches, handler)

    # Compute node caches
    act_cache = compute_node_act_cache(indexer.src_index, caches.act_cache)
    grad_cache = compute_node_grad_cache(indexer.dest_index, caches.grad_cache)

    # Test tensor shapes
    for cache in [act_cache, grad_cache]:
        batch_size, n_pos, _, d_model = cache.shape
        assert batch_size == handler.get_batch_size()
        assert n_pos == handler.get_n_pos()
        assert d_model == ts_model.cfg.d_model
