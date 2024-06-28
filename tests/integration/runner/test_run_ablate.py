from sae_eap.core.types import Model
from sae_eap.runner import run_ablation, run_attribution
from sae_eap.graph.build import build_graph
from sae_eap.prune import ThresholdEdgePruner, PruningPipeline, DeadNodePruner

from tests.integration.attribute.helpers import make_single_prompt_handler


def make_pruner():
    return (
        PruningPipeline()
        .add_pruner(ThresholdEdgePruner(1.0))
        .add_pruner(DeadNodePruner())
    )


def test_run_ablation(ts_model: Model):
    graph = build_graph(ts_model.cfg)
    handler = make_single_prompt_handler(ts_model)
    scores_dict = run_attribution(ts_model, graph, handler)

    model_graph = graph.copy()
    make_pruner().prune(graph, scores_dict)
    circuit_graph = graph
    faithfulness = run_ablation(ts_model, circuit_graph, model_graph, handler)
