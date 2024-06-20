from sae_eap.core.types import Model
from sae_eap.attribute import (
    compute_activations_and_gradients_simple,
)
from sae_eap.graph.build import build_graph
from sae_eap.utils import DeviceManager

from tests.unit.attribute.helpers import make_single_prompt_handler


def test_compute_activations_and_gradients_simple_has_correct_tensor_shape(
    ts_model: Model,
):
    with DeviceManager.instance().use_device("cpu"):
        graph = build_graph(ts_model.cfg)
        handler = make_single_prompt_handler(ts_model)
        activations, gradients = compute_activations_and_gradients_simple(
            ts_model, graph, handler
        )

    assert activations.shape == (1, 2, len(graph.src_nodes), ts_model.cfg.d_model)
    assert gradients.shape == (1, 2, len(graph.dest_nodes), ts_model.cfg.d_model)
