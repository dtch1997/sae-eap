# type: ignore

import torch
from jaxtyping import Float

from sae_eap.core.types import HookName, ForwardHook
from sae_eap.utils import DeviceManager

# NOTE: variadic type annotation
# There can be 1 or more dimensions after the first two
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

    def zero(self):
        """Zero out all the tensors in the cache."""
        for key in self.keys():
            self[key].zero_()

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


def make_cache_setter_hook(
    cache: CacheDict, hook_name: HookName, add: bool = True
) -> ForwardHook:
    """Factory function for TransformerLens hooks that cache a value."""

    def hook_fn(activations, hook) -> None:
        assert hook_name == hook.name, f"Expected {hook_name}, got {hook.name}"
        assert hook_name not in cache, f"Hook {hook_name} already in cache."

        acts: CacheTensor = activations.detach()
        cache[hook.name] = init_cache_tensor(acts.size(), dtype=acts.dtype)
        if add:
            cache[hook.name] += acts
        else:
            cache[hook.name] -= acts

    return ForwardHook(hook_name, hook_fn)
