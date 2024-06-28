from tqdm import tqdm
from typing import Iterator

from sae_eap.cache import (
    init_cache_tensor,
    CacheDict,
    make_cache_adder_hooks_for_unique_hook_names,
)
from sae_eap.core.types import Model
from sae_eap.graph import TensorGraph
from sae_eap.graph.index import TensorGraphIndexer
from sae_eap.data.handler import BatchHandler
from sae_eap.attribute import (
    EdgeAttributionScores,
    make_cache_hooks_and_dicts,
    compute_model_caches,
    compute_node_act_cache,
    compute_node_grad_cache,
    compute_attribution_scores,
)
from sae_eap.ablate import (
    AblateSetting,
    make_edge_ablate_hooks,
    get_clean_and_ablate_input,
)


def run_attribution(
    model: Model,
    graph: TensorGraph,
    iter_batch_handler: Iterator[BatchHandler] | BatchHandler,
    *,
    aggregation="sum",
    quiet=False,
) -> EdgeAttributionScores:
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


def run_ablation(
    model: Model,
    circuit_graph: TensorGraph,
    model_graph: TensorGraph,
    iter_batch_handler: Iterator[BatchHandler] | BatchHandler,
    *,
    setting: AblateSetting = "denoising",
) -> list[float]:
    if isinstance(iter_batch_handler, BatchHandler):
        iter_batch_handler = iter([iter_batch_handler])

    clean_input, ablate_input = get_clean_and_ablate_input(setting)

    # Make caches and setter hooks
    src_hook_name_set = set(node.hook for node in model_graph.src_nodes)
    ablate_cache = CacheDict()
    ablate_cache_setter_hooks = make_cache_adder_hooks_for_unique_hook_names(
        src_hook_name_set, ablate_cache, add=True
    )
    clean_cache = CacheDict()
    clean_cache_setter_hooks = make_cache_adder_hooks_for_unique_hook_names(
        src_hook_name_set, clean_cache, add=False
    )

    # Make the hooks to ablate edges in a circuit
    edge_ablate_hooks = make_edge_ablate_hooks(
        circuit_graph=circuit_graph,
        model_graph=model_graph,
        clean_cache=clean_cache,
        ablate_cache=ablate_cache,
    )

    faithfulnesses = []
    for handler in iter_batch_handler:
        # Populate the clean cache
        # Calculate clean metric
        with model.hooks(fwd_hooks=clean_cache_setter_hooks):  # type: ignore
            clean_logits = handler.get_logits(model, input=clean_input)
            clean_metric = handler.get_metric(clean_logits)

        # Populate the ablate cache
        # Calculate ablated metric
        with model.hooks(fwd_hooks=ablate_cache_setter_hooks):  # type: ignore
            fully_ablated_logits = handler.get_logits(model, input=ablate_input)
            fully_ablated_metric = handler.get_metric(fully_ablated_logits)

        # Run the model with intervention hooks
        # Calculate circuit ablated metric
        with model.hooks(fwd_hooks=edge_ablate_hooks):  # type: ignore
            circuit_ablated_logits = handler.get_logits(model, input=clean_input)
            circuit_ablated_metric = handler.get_metric(circuit_ablated_logits)

        # Faithfulness computed as (circuit_ablated - fully_ablated) / (clean - fully_ablated)
        faithfulness = (circuit_ablated_metric - fully_ablated_metric).mean() / (
            clean_metric - fully_ablated_metric
        ).mean()
        faithfulness = faithfulness.item()
        faithfulnesses.append(faithfulness)

    return faithfulnesses
