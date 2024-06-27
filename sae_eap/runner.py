from tqdm import tqdm
from typing import Iterator

from sae_eap.cache import init_cache_tensor
from sae_eap.core.types import Model
from sae_eap.graph import TensorGraph
from sae_eap.graph.index import TensorGraphIndexer
from sae_eap.data.handler import BatchHandler
from sae_eap.attribute import (
    AttributionScores,
    make_cache_hooks_and_dicts,
    compute_model_caches,
    compute_node_act_cache,
    compute_node_grad_cache,
    compute_attribution_scores,
)


def run_attribution(
    model: Model,
    graph: TensorGraph,
    iter_batch_handler: Iterator[BatchHandler] | BatchHandler,
    *,
    aggregation="sum",
    quiet=False,
) -> AttributionScores:
    if isinstance(iter_batch_handler, BatchHandler):
        iter_batch_handler = iter([iter_batch_handler])

    # Initialize the cache tensor
    indexer = TensorGraphIndexer(graph)
    scores_cache = init_cache_tensor(
        shape=(len(graph.src_nodes), len(graph.dest_nodes))
    )

    # Compute the attribution scores
    total_items = 0
    for handler in tqdm(iter_batch_handler, disable=quiet):
        total_items += handler.get_batch_size()
        hooks, caches = make_cache_hooks_and_dicts(graph)
        model_caches = compute_model_caches(model, hooks, caches, handler)
        node_act_cache = compute_node_act_cache(
            indexer.src_index, model_caches.act_cache
        )
        node_grad_cache = compute_node_grad_cache(
            indexer.dest_index, model_caches.grad_cache
        )
        # TODO: Add strategy for integrated gradients.
        scores = compute_attribution_scores(
            node_act_cache, node_grad_cache, model.cfg, aggregation=aggregation
        )
        scores_cache += scores

    scores_cache /= total_items
    scores_cache = scores_cache.cpu().numpy()

    scores_dict = {}
    for edge in tqdm(graph.edges, total=len(graph.edges), disable=quiet):
        score = scores_cache[
            indexer.get_src_index(edge.src), indexer.get_dest_index(edge.dest)
        ]
        scores_dict[edge.name] = score

    return scores_dict


def run_ablation(model: Model, circuit_graph: TensorGraph, handler: BatchHandler):
    pass
