# type: ignore

import torch


from typing import Iterator, NamedTuple
from jaxtyping import Float

from tqdm import tqdm
from einops import einsum
from sae_eap.core.types import TLForwardHook, TLBackwardHook, HookName
from sae_eap.utils import DeviceManager
from sae_eap.graph import TensorGraph
from sae_eap.graph.index import TensorGraphIndexer, TensorNodeIndex
from sae_eap.data.handler import BatchHandler
from transformer_lens import HookedTransformer, HookedTransformerConfig

CacheTensor = Float[torch.Tensor, "batch pos n_node d_model"]


# Define a hook function to store a tensor
# When doing clean forward pass, we'll want to add activations to the tensor
# When doing corrupted forward pass, we'll want to subtract activations from the tensor
def get_cache_hook(cache: CacheTensor, index: TensorNodeIndex, add: bool = True):
    """Factory function for TransformerLens hooks that cache a value."""

    def hook_fn(activations, hook):
        acts: Float[torch.Tensor, "batch pos d_model"] = activations.detach()
        try:
            if add:
                cache[:, :, index] += acts
            else:
                cache[:, :, index] -= acts
        except RuntimeError as e:
            # Some useful debugging information
            print(hook.name, cache.size(), acts.size())
            raise e

    return hook_fn


def init_cache_tensor(
    shape: tuple[int, ...],
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
):
    """Initialize a cache tensor."""
    if device is None:
        device = DeviceManager.instance().get_device()
    return torch.zeros(
        shape,
        device=device,
        dtype=dtype,
    )


CacheHooks = NamedTuple(
    "CacheHooks",
    [
        ("fwd_hooks_clean", list[tuple[HookName, TLForwardHook]]),
        ("fwd_hooks_corrupt", list[tuple[HookName, TLForwardHook]]),
        ("bwd_hooks_clean", list[tuple[HookName, TLBackwardHook]]),
    ],
)

CacheTensors = NamedTuple(
    "CacheTensors",
    [
        ("act_cache", CacheTensor),
        ("grad_cache", CacheTensor),
    ],
)


def make_hooks_and_tensors(
    graph: TensorGraph,
    batch_size: int = 1,
    n_token: int = 1,
) -> tuple[CacheHooks, CacheTensors]:
    """Make hooks and tensors to do attribution patching.

    Args:
        graph: The graph containing all nodes and edges in the model.
        batch_size: The batch size.
        n_token: The number of tokens.

    Returns:
        hooks:
            - fwd_hooks_clean: Forward hooks to store clean activations.
            - fwd_hooks_corrupt: Forward hooks to store corrupted activations.
            - bwd_hooks_clean: Backward hooks to store clean gradients.
        tensors:
            - activation_delta_cache: A tensor where activations are stored by the forward hooks.
            - gradient_cache: A tensor where gradients are stored by the backward hooks.
    """

    # Initialization
    fwd_hooks_clean = []
    fwd_hooks_corrupt = []
    bwd_hooks_clean = []

    indexer = TensorGraphIndexer(graph)

    activation_delta_cache: Float[torch.Tensor, "batch pos n_src d_model"] = (
        init_cache_tensor(
            shape=(batch_size, n_token, len(graph.src_nodes), graph.cfg.d_model)
        )
    )
    gradient_cache: Float[torch.Tensor, "batch pos n_input d_model"] = (
        init_cache_tensor(
            shape=(batch_size, n_token, len(graph.dest_nodes), graph.cfg.d_model)
        )
    )

    # Populate the hooks and tensors.
    for node in graph.src_nodes:
        # Forward clean hook
        index = indexer.get_src_index(node)
        hook = get_cache_hook(activation_delta_cache, index, add=True)
        fwd_hooks_clean.append((node.hook, hook))

        # Forward corrupt hook
        index = indexer.get_src_index(node)
        hook = get_cache_hook(activation_delta_cache, index, add=False)
        fwd_hooks_corrupt.append((node.hook, hook))

    for node in graph.dest_nodes:
        # Backward clean hook
        index = indexer.get_dest_index(node)
        hook = get_cache_hook(gradient_cache, index, add=True)
        bwd_hooks_clean.append((node.hook, hook))

    hooks = CacheHooks(fwd_hooks_clean, fwd_hooks_corrupt, bwd_hooks_clean)
    tensors = CacheTensors(activation_delta_cache, gradient_cache)
    return hooks, tensors


def compute_activations_and_gradients_simple(
    model: HookedTransformer,
    graph: TensorGraph,
    handler: BatchHandler,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple computation of activations."""

    # Make hooks and tensors
    (
        (fwd_hooks_clean, fwd_hooks_corrupt, bwd_hooks_clean),
        (activation_difference, gradients),
    ) = make_hooks_and_tensors(
        graph, batch_size=handler.get_batch_size(), n_token=handler.get_n_pos()
    )

    # Store the activations for the corrupt inputs
    with model.hooks(fwd_hooks=fwd_hooks_corrupt):
        handler.get_logits(model, input="corrupt")

    # Store the activations and gradients for the clean inputs
    with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks_clean):
        logits = handler.get_logits(model, input="clean")
        metric = handler.get_metric(logits)
        metric.backward()

    return activation_difference, gradients


# TODO: integrated gradients

allowed_aggregations = {"sum", "mean", "l2"}


def compute_attribution_scores(
    activation_differences: CacheTensor,
    gradients: CacheTensor,
    cfg: HookedTransformerConfig,
    *,
    aggregation: str = "sum",
):
    """Compute the attribution scores using the activation differences and gradients."""
    if aggregation not in allowed_aggregations:
        raise ValueError(
            f"aggregation must be in {allowed_aggregations}, but got {aggregation}"
        )

    scores = einsum(
        activation_differences,
        gradients,
        "batch pos n_output d_model, batch pos n_input d_model -> n_output n_input",
    )

    if aggregation == "mean":
        scores /= cfg.d_model
    elif aggregation == "l2":
        scores = torch.linalg.vector_norm(scores, ord=2, dim=-1)

    return scores


def attribute(
    model: HookedTransformer,
    graph: TensorGraph,
    iter_batch_handler: Iterator[BatchHandler] | BatchHandler,
    *,
    aggregation="sum",
    quiet=False,
):
    if isinstance(iter_batch_handler, BatchHandler):
        iter_batch_handler = iter([iter_batch_handler])

    # Initialize the cache tensor
    indexer = TensorGraphIndexer(graph)
    scores_cache = init_cache_tensor(
        shape=(len(graph.src_nodes), len(graph.dest_nodes))
    )

    # Compute the attribution scores
    total_items = 0
    iter_batch_handler = tqdm(iter_batch_handler, disable=quiet)
    for handler in iter_batch_handler:
        total_items += handler.get_batch_size()
        # TODO: Add strategy for integrated gradients.
        activation_differences, gradients = compute_activations_and_gradients_simple(
            model, graph, handler
        )
        scores = compute_attribution_scores(
            activation_differences, gradients, model.cfg, aggregation=aggregation
        )
    scores_cache += scores
    scores_cache /= total_items
    scores_cache = scores_cache.cpu().numpy()

    # Update the scores in the graph
    for edge in tqdm(graph.edges, total=len(graph.edges), disable=quiet):
        score = scores_cache[
            indexer.get_src_index(edge.src), indexer.get_dest_index(edge.dest)
        ]
        graph.set_edge_info(edge, {"score": score})
