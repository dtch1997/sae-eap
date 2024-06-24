# type: ignore

import torch


from typing import Iterator, NamedTuple
from jaxtyping import Float

from tqdm import tqdm
from einops import einsum
from sae_eap.core.types import TLForwardHook, TLBackwardHook, HookName
from sae_eap.utils import DeviceManager
from sae_eap.graph import TensorGraph
from sae_eap.graph.node import SrcNode, DestNode
from sae_eap.graph.index import TensorGraphIndexer, TensorNodeIndex
from sae_eap.data.handler import BatchHandler
from transformer_lens import HookedTransformer, HookedTransformerConfig

# NOTE: variadic type annotation
# There can be arbitrarily many dimensiosn after the first two
CacheTensor = Float[torch.Tensor, "batch pos * d_model"]


class CacheDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: HookName) -> CacheTensor:
        return super().__getitem__(key)

    def __setitem__(self, key: HookName, value: CacheTensor):
        super().__setitem__(key, value)

    def __repr__(self):
        return f"CacheDict({super().__repr__()})"

    @property
    def batch_size(self) -> int:
        return next(iter(self.values())).size(0)

    @property
    def n_pos(self) -> int:
        return next(iter(self.values())).size(1)

    @property
    def d_model(self) -> int:
        return next(iter(self.values())).size(-1)


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


# Define a hook function to store a tensor
# When doing clean forward pass, we'll want to add activations to the tensor
# When doing corrupted forward pass, we'll want to subtract activations from the tensor
def get_cache_hook(cache: CacheDict, add: bool = True):
    """Factory function for TransformerLens hooks that cache a value."""

    def hook_fn(activations, hook):
        acts: CacheTensor = activations.detach()
        if hook.name not in cache:
            cache[hook.name] = init_cache_tensor(acts.size(), dtype=acts.dtype)
        try:
            if add:
                cache[hook.name] += acts
            else:
                cache[hook.name] -= acts
        except RuntimeError as e:
            # Some useful debugging information
            print(hook.name, cache.size(), acts.size())
            raise e

    return hook_fn


CacheHooks = NamedTuple(
    "CacheHooks",
    [
        ("fwd_hooks_clean", list[tuple[HookName, TLForwardHook]]),
        ("fwd_hooks_corrupt", list[tuple[HookName, TLForwardHook]]),
        ("bwd_hooks_clean", list[tuple[HookName, TLBackwardHook]]),
    ],
)

CacheDicts = NamedTuple(
    "CacheDicts",
    [
        ("act_cache", CacheDict),
        ("grad_cache", CacheDict),
    ],
)


def make_cache_hooks_and_dicts(
    graph: TensorGraph,
) -> tuple[CacheHooks, CacheDicts]:
    """Make hooks and tensors to cache model activations and gradients.

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

    activation_delta_cache = CacheDict()
    gradient_cache = CacheDict()

    # Populate the hooks and tensors.
    for node in graph.src_nodes:
        # Forward clean hook
        hook = get_cache_hook(activation_delta_cache, add=True)
        fwd_hooks_clean.append((node.hook, hook))

        # Forward corrupt hook
        hook = get_cache_hook(activation_delta_cache, add=False)
        fwd_hooks_corrupt.append((node.hook, hook))

    for node in graph.dest_nodes:
        # Backward clean hook
        hook = get_cache_hook(gradient_cache, add=True)
        bwd_hooks_clean.append((node.hook, hook))

    hooks = CacheHooks(fwd_hooks_clean, fwd_hooks_corrupt, bwd_hooks_clean)
    caches = CacheDicts(activation_delta_cache, gradient_cache)
    return hooks, caches


def get_model_caches(
    model: HookedTransformer,
    graph: TensorGraph,
    handler: BatchHandler,
) -> CacheDicts:
    """Simple computation of activations."""

    # Make hooks and tensors
    hooks, caches = make_cache_hooks_and_dicts(graph)

    # Store the activations for the corrupt inputs
    with model.hooks(fwd_hooks=hooks.fwd_hooks_corrupt):
        handler.get_logits(model, input="corrupt")

    # Store the activations and gradients for the clean inputs
    with model.hooks(fwd_hooks=hooks.fwd_hooks_clean, bwd_hooks=hooks.bwd_hooks_clean):
        logits = handler.get_logits(model, input="clean")
        metric = handler.get_metric(logits)
        metric.backward()

    return caches


def compute_node_act_cache(
    node_index: dict[SrcNode, TensorNodeIndex],
    model_act_cache: CacheDict,
) -> CacheTensor:
    node_act_cache = init_cache_tensor(
        shape=(
            model_act_cache.batch_size,
            model_act_cache.n_pos,
            len(node_index),
            model_act_cache.d_model,
        )
    )
    for node, index in node_index.items():
        model_act = model_act_cache[node.hook]
        node_act = node.get_act(model_act)
        assert len(node_act.shape) == 3
        node_act_cache[:, :, index, :] = node_act
    return node_act_cache


def compute_node_grad_cache(
    node_index: dict[DestNode, TensorNodeIndex],
    model_grad_cache: CacheDict,
) -> CacheTensor:
    node_grad_cache = init_cache_tensor(
        shape=(
            model_grad_cache.batch_size,
            model_grad_cache.n_pos,
            len(node_index),
            model_grad_cache.d_model,
        )
    )
    for node, index in node_index.items():
        model_grad = model_grad_cache[node.hook]
        node_grad_cache[:, :, index] = node.get_grad(model_grad)
    return node_grad_cache


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
        model_caches = get_model_caches(model, graph, handler)
        node_act_cache = compute_node_act_cache(
            indexer.src_index, model_caches.act_cache
        )
        node_grad_cache = compute_node_grad_cache(
            indexer.dest_index, model_caches.grad_cache
        )

        scores = compute_attribution_scores(
            node_act_cache, node_grad_cache, model.cfg, aggregation=aggregation
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
