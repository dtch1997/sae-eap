from sae_eap.core.types import Model
from sae_eap.runner import run_attribution
from sae_eap.graph.build import build_graph

from tests.integration.attribute.helpers import make_single_prompt_handler


def test_attribute(ts_model: Model):
    graph = build_graph(ts_model.cfg)
    handler = make_single_prompt_handler(ts_model)
    scores_dict = run_attribution(ts_model, graph, handler)
    assert len(scores_dict) == len(graph.edges)
