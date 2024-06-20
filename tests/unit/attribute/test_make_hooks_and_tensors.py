import pytest

from sae_eap.core.types import Model
from sae_eap.attribute import (
    make_hooks_and_tensors,
)
from sae_eap.graph.build import build_graph
from sae_eap.utils import DeviceManager

from tests.unit.attribute.helpers import make_single_prompt_handler


def test_hook_lengths_and_tensor_shapes(ts_model: Model):
    with DeviceManager.instance().use_device("cpu"):
        graph = build_graph(ts_model.cfg)
        handler = make_single_prompt_handler(ts_model)
        hooks, tensors = make_hooks_and_tensors(
            graph, handler.get_batch_size(), handler.get_n_pos()
        )

    # Test hook lengths
    assert len(hooks.fwd_hooks_clean) == len(graph.src_nodes)
    assert len(hooks.fwd_hooks_corrupt) == len(graph.src_nodes)
    assert len(hooks.bwd_hooks_clean) == len(graph.dest_nodes)

    # Test tensor shapes
    assert tensors.act_cache.shape == (
        handler.get_batch_size(),
        handler.get_n_pos(),
        len(graph.src_nodes),
        ts_model.cfg.d_model,
    )

    assert tensors.grad_cache.shape == (
        handler.get_batch_size(),
        handler.get_n_pos(),
        len(graph.dest_nodes),
        ts_model.cfg.d_model,
    )


@pytest.mark.skip("Not implemented")
def test_forward_hooks_work(ts_model: Model):
    """Test that running model with forward hooks modifies the act cache."""
    raise NotImplementedError


@pytest.mark.skip("Not implemented")
def test_backward_hooks_work(ts_model: Model):
    """Test that running model with forward hooks modifies the act cache."""
    raise NotImplementedError
