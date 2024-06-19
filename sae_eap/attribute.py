# type: ignore
import torch

from typing import Callable, Iterator
from jaxtyping import Float

from tqdm import tqdm
from einops import einsum
from sae_eap.core.types import TLForwardHook, TLBackwardHook
from sae_eap.utils import get_device, get_npos_and_input_lengths
from sae_eap.graph import Graph
from sae_eap.graph.index import GraphIndexer, Index
from transformer_lens import HookedTransformer, HookedTransformerConfig

CacheTensor = Float[torch.Tensor, "batch pos n_node d_model"]


# Define a hook function to store a tensor
# When doing clean forward pass, we'll want to add activations to the tensor
# When doing corrupted forward pass, we'll want to subtract activations from the tensor
def get_cache_hook(cache: CacheTensor, index: Index, add: bool = True):
    """Factory function for TransformerLens hooks that cache a value."""

    def hook_fn(activations, hook):
        acts = activations.detach()
        try:
            if add:
                cache[index] += acts
            else:
                cache[index] -= acts
        except RuntimeError as e:
            # Some useful debugging information
            print(hook.name, cache.size(), acts.size())
            raise e

    return hook_fn


def init_cache_tensor(
    shape: tuple[int, ...],
    device: str = get_device(),
    dtype: torch.dtype = torch.float32,
):
    """Initialize a cache tensor."""
    return torch.zeros(
        shape,
        device=device,
        dtype=dtype,
    )


def make_hooks_and_tensors(
    graph: Graph,
    batch_size: int = 1,
    n_token: int = 1,
) -> tuple[
    tuple[list[TLForwardHook], list[TLForwardHook], list[TLBackwardHook]],
    tuple[CacheTensor, CacheTensor],
]:
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

    indexer = GraphIndexer(graph)

    activation_delta_cache: Float[torch.Tensor, "batch pos n_output d_model"] = (
        init_cache_tensor(
            shape=(batch_size, n_token, indexer.n_outputs, graph.cfg.d_model)
        )
    )
    gradient_cache: Float[torch.Tensor, "batch pos n_input d_model"] = (
        init_cache_tensor(
            shape=(batch_size, n_token, indexer.n_inputs, graph.cfg.d_model)
        )
    )

    # Populate the hooks and tensors.
    for node in graph.nodes:
        # Forward clean hook
        index = indexer.get_output_index(node)
        hook = get_cache_hook(activation_delta_cache, index, add=True)
        fwd_hooks_clean.append(hook)

        # Forward corrupt hook
        index = indexer.get_output_index(node)
        hook = get_cache_hook(activation_delta_cache, index, add=False)
        fwd_hooks_corrupt.append(hook)

        # Backward clean hook
        index = indexer.get_input_index(node)
        hook = get_cache_hook(gradient_cache, index, add=True)
        bwd_hooks_clean.append(hook)

    hooks = (fwd_hooks_clean, fwd_hooks_corrupt, bwd_hooks_clean)
    tensors = (activation_delta_cache, gradient_cache)
    return hooks, tensors


def compute_activations_and_gradients_simple(
    model: HookedTransformer,
    graph: Graph,
    clean_inputs: list[str],
    corrupt_inputs: list[str],
    metric: Callable[[torch.Tensor], torch.Tensor],
    labels,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple computation of activations."""
    batch_size = len(clean_inputs)
    n_pos, input_lengths = get_npos_and_input_lengths(model, clean_inputs)

    (
        (fwd_hooks_clean, fwd_hooks_corrupt, bwd_hooks_clean),
        (activation_difference, gradients),
    ) = make_hooks_and_tensors(graph, batch_size, n_pos)

    # Store the activations for the corrupt inputs
    with model.hooks(fwd_hooks=fwd_hooks_corrupt):
        corrupted_logits = model(corrupt_inputs)

    # Store the activations and gradients for the clean inputs
    with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks_clean):
        logits = model(clean_inputs)
        metric_value = metric(logits, corrupted_logits, input_lengths, labels)
        metric_value.backward()

    return activation_difference, gradients


def compute_activations_and_gradients_ig(
    model: HookedTransformer,
    graph: Graph,
    clean_inputs: list[str],
    corrupted_inputs: list[str],
    metric: Callable[[torch.Tensor], torch.Tensor],
    labels,
    steps=30,
):
    """Compute the attribution using integrated gradients."""

    raise NotImplementedError("This function is not implemented yet.")

    batch_size = len(clean_inputs)
    n_pos, input_lengths = get_npos_and_input_lengths(model, clean_inputs)

    (
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks),
        (activation_difference, gradients),
    ) = make_hooks_and_tensors(graph, batch_size, n_pos)

    with torch.inference_mode():
        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            _ = model(corrupted_inputs)

        input_activations_corrupted = activation_difference[
            :, :, graph.forward_index(graph.nodes["input"])
        ].clone()

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_inputs)

        input_activations_clean = (
            input_activations_corrupted
            - activation_difference[:, :, graph.forward_index(graph.nodes["input"])]
        )

    def input_interpolation_hook(k: int):
        def hook_fn(activations, hook):
            new_input = input_activations_corrupted + (k / steps) * (
                input_activations_clean - input_activations_corrupted
            )
            new_input.requires_grad = True
            return new_input

        return hook_fn

    total_steps = 0
    for step in range(1, steps + 1):
        total_steps += 1
        with model.hooks(
            fwd_hooks=[(graph.nodes["input"].out_hook, input_interpolation_hook(step))],
            bwd_hooks=bwd_hooks,
        ):
            logits = model(clean_inputs)
            metric_value = metric(logits, clean_logits, input_lengths, labels)
            metric_value.backward()

    gradients = gradients / total_steps

    return activation_difference, gradients


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
    graph: Graph,
    dataloader: Iterator[tuple[list[str], list[str], torch.Tensor]],
    metric: Callable[[torch.Tensor], torch.Tensor],
    *,
    aggregation="sum",
    quiet=False,
):
    # Initialize the cache tensor
    indexer = GraphIndexer(graph)
    scores_cache = init_cache_tensor(shape=(indexer.n_outputs, indexer.n_inputs))

    # Compute the attribution scores
    total_items = 0
    dataloader = tqdm(dataloader, disable=quiet)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        activation_differences, gradients = compute_activations_and_gradients_simple(
            model, graph, clean, corrupted, metric, label
        )
        scores = compute_attribution_scores(
            activation_differences, gradients, model.cfg, aggregation=aggregation
        )
        scores_cache += scores
    scores_cache /= total_items
    scores_cache = scores_cache.cpu().numpy()

    # Update the scores in the graph
    for edge in tqdm(graph.edges, total=len(graph.edges), disable=quiet):
        # TODO: get the score
        score = 0
        graph.set_edge_info(edge, {"score": score})
