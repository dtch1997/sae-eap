import torch

from sae_eap.core.types import Model
from sae_eap.attribute import (
    make_cache_hooks_and_dicts,
)
from sae_eap.graph.build import build_graph


def test_make_cache_hooks_and_dicts_model_cache_matches_transformer_lens_cache(
    ts_model: Model,
):
    model = ts_model
    graph = build_graph(ts_model.cfg)
    hooks, caches = make_cache_hooks_and_dicts(graph)
    act_cache, _ = caches

    # Test the fwd hook clean
    with model.hooks(fwd_hooks=hooks.fwd_hooks_clean):  # type: ignore
        _, tl_cache = model.run_with_cache("Hello, world")

    for our_key, our_value in act_cache.items():
        assert our_key in tl_cache
        tl_value = tl_cache[our_key]
        assert our_value.shape == tl_value.shape
        # NOTE: forward hooks clean subtract the act from the cache
        assert torch.allclose(our_value, -1 * tl_value)

    # Test the fwd hook corrupt
    act_cache.clear()
    with model.hooks(fwd_hooks=hooks.fwd_hooks_corrupt):  # type: ignore
        _, tl_cache = model.run_with_cache("Hello, world")

    for our_key, our_value in act_cache.items():
        assert our_key in tl_cache
        tl_value = tl_cache[our_key]
        assert our_value.shape == tl_value.shape
        assert torch.allclose(our_value, tl_value)
