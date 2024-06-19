import pytest

from sae_eap.attribute import compute_activations_and_gradients_simple
from sae_eap.graph.build import build_graph
from sae_eap.data.handler import SinglePromptHandler


def make_single_prompt_handler(ts_model):
    return SinglePromptHandler(
        model=ts_model,
        clean_prompt="clean",
        corrupt_prompt="dirty",
        answer=" answer",
        wrong_answer=" wrong",
    )


@pytest.mark.xfail
def test_compute_activations_and_gradients_simple(ts_model):
    # Test the function with a simple example
    graph = build_graph(ts_model.cfg)
    single_prompt_handler = make_single_prompt_handler(ts_model)
    activations, gradients = compute_activations_and_gradients_simple(
        ts_model, graph, single_prompt_handler
    )
