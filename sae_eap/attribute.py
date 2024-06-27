import torch


from typing import NamedTuple
from einops import einsum

from sae_eap import utils
from sae_eap.core.types import ForwardHook, BackwardHook
from sae_eap.graph import TensorGraph
from sae_eap.graph.node import SrcNode, DestNode
from sae_eap.graph.index import TensorNodeIndex
from sae_eap.cache import (
    CacheDict,
    CacheTensor,
    init_cache_tensor,
    make_cache_setter_hook,
)
from sae_eap.data.handler import BatchHandler

from transformer_lens import HookedTransformer, HookedTransformerConfig

CacheHooks = NamedTuple(
    "CacheHooks",
    [
        ("fwd_hooks_clean", list[ForwardHook]),
        ("fwd_hooks_corrupt", list[ForwardHook]),
        ("bwd_hooks_clean", list[BackwardHook]),
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
        hook = make_cache_setter_hook(activation_delta_cache, node.hook, add=False)
        fwd_hooks_clean.append(hook)

        # Forward corrupt hook
        hook = make_cache_setter_hook(activation_delta_cache, node.hook, add=True)
        fwd_hooks_corrupt.append(hook)

    for node in graph.dest_nodes:
        # Backward clean hook
        hook = make_cache_setter_hook(gradient_cache, node.hook, add=True)
        bwd_hooks_clean.append(hook)

    hooks = CacheHooks(fwd_hooks_clean, fwd_hooks_corrupt, bwd_hooks_clean)
    caches = CacheDicts(activation_delta_cache, gradient_cache)
    return hooks, caches


def compute_model_caches(
    model: HookedTransformer,
    hooks: CacheHooks,
    caches: CacheDicts,
    handler: BatchHandler,
) -> CacheDicts:
    """Simple computation of activations."""

    # Store the activations for the corrupt inputs
    with model.hooks(fwd_hooks=hooks.fwd_hooks_corrupt):  # type: ignore
        handler.get_logits(model, input="corrupt")

    # Store the activations and gradients for the clean inputs
    with model.hooks(fwd_hooks=hooks.fwd_hooks_clean, bwd_hooks=hooks.bwd_hooks_clean):  # type: ignore
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
        node_grad = node.get_grad(model_grad)
        assert len(node_grad.shape) == 3
        node_grad_cache[:, :, index] = node_grad
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


AttributionScores = dict[str, float]


def save_attribution_scores(
    scores_dict: AttributionScores,
    savedir: str,
    filename: str = "attrib_scores",
):
    """Save the attribution scores to a pickle file."""
    utils.save_obj_as_pickle(scores_dict, savedir, filename)


def load_attribution_scores(
    savedir: str,
    filename: str = "attrib_scores",
) -> AttributionScores:
    """Load the attribution scores from a pickle file."""
    obj = utils.load_obj_from_pickle(savedir, filename)
    assert isinstance(obj, dict), f"Expected dict, got {type(obj)}"
    return obj
