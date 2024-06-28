"""Defines lightweight caching utilities.

Here, we opt to create our own caching setup to:
- support multiple hooks writing to the same cache dict.
"""

# type: ignore

import torch
from jaxtyping import Float

from typing import Iterable
from collections import namedtuple
from sae_eap.core.types import HookName
from sae_eap.utils import DeviceManager

# NOTE: variadic type annotation
# There can be 1 or more dimensions after the first two
CacheTensor = Float[torch.Tensor, "batch pos * d_model"]
CacheHook = namedtuple("CacheHook", ["hook_name", "hook_fn"])


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


def make_cache_adder_hook(
    cache: CacheDict, hook_name: HookName, add: bool = True
) -> CacheHook:
    """Factory function for TransformerLens hooks that adds a value to a cache"""

    def hook_fn(activations, hook) -> None:
        if hook_name != hook.name:
            raise RuntimeError(f"Expected {hook_name}, got {hook.name}")

        acts: CacheTensor = activations.detach()
        cache[hook.name] = init_cache_tensor(acts.size(), dtype=acts.dtype)
        if add:
            cache[hook.name] += acts
        else:
            cache[hook.name] -= acts

    return CacheHook(hook_name, hook_fn)


def make_cache_adder_hooks_for_unique_hook_names(
    hook_names: Iterable[HookName], cache: CacheDict, add: bool = True
) -> list[CacheHook]:
    """A utility function to make one adder hook per hook point.

    Deduplicates the nodes by hook name and makes a hook for each unique hook name.
    """
    deduped_hook_names = set(hook_names)
    hooks = []
    for hook_name in deduped_hook_names:
        hook = make_cache_adder_hook(cache, hook_name, add=add)
        hooks.append(hook)
    return hooks
