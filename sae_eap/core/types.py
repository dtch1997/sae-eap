import torch
import transformer_lens as tl

from jaxtyping import Float, Int
from typing import Protocol, Callable, Literal
from transformer_lens.hook_points import HookPoint

Device = Literal["cpu", "cuda", "mps"]
LayerIndex = int
FeatureIndex = int
TokenIndex = int

HookName = str
HookNameFilterFn = Callable[[HookName], bool]

# Torch types
Tokens = Int[torch.Tensor, "batch seq"]
Logits = Float[torch.Tensor, "batch seq d_vocab"]
Model = tl.HookedTransformer
Metric = Float[torch.Tensor, " ()"]
MetricFn = Callable[[Model, Tokens], Metric]

# SAE types
SaeFamily = Literal["res-jb", "att-kk", "tres-dc"]


class TransformerLensForwardHook(Protocol):
    def __call__(self, orig: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        raise NotImplementedError


class TransformerLensBackwardHook(Protocol):
    def __call__(self, orig: torch.Tensor, hook: HookPoint) -> tuple[torch.Tensor]:
        raise NotImplementedError
