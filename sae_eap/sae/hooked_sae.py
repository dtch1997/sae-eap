from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
import transformer_lens as tl
from transformer_lens.hook_points import HookPoint

import pprint
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np

from sae_eap.utils import get_device


@dataclass
class HookedSAEConfig:
    """
    Configuration class to store the configuration of a HookedSAE model.

    Args:
        d_sae (int): The size of the dictionary.
        d_in (int): The dimension of the input activations.
        hook_name (str): The hook name of the activation the SAE was trained on (eg. blocks.0.attn.hook_z)
        use_error_term (bool): Whether to use the error term in the loss function. Defaults to False.
        dtype (torch.dtype, *optional*): The SAE's dtype. Defaults to torch.float32.
        seed (int, *optional*): The seed to use for the SAE.
            Used to set sources of randomness (Python, PyTorch and
            NumPy) and to initialize weights. Defaults to None. We recommend setting a seed, so your experiments are reproducible.
        device(str): The device to use for the SAE. Defaults to 'cuda' if
            available, else 'cpu'.
    """

    d_sae: int
    d_in: int
    hook_name: str
    use_error_term: bool = False
    dtype: torch.dtype = torch.float32
    seed: Optional[int] = None
    device: Optional[str] = None

    def __post_init__(self):
        if self.seed is not None:
            self.set_seed_everywhere(self.seed)

        if self.device is None:
            self.device = get_device()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> HookedSAEConfig:
        """
        Instantiates a `HookedSAEConfig` from a Python dictionary of
        parameters.
        """
        return cls(**config_dict)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return "HookedSAEConfig:\n" + pprint.pformat(self.to_dict())

    def set_seed_everywhere(self, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


class HookedSAE(tl.hook_points.HookedRootModule):
    """Hooked SAE.

    Implements a standard SAE with a TransformerLens hooks for SAE activations

    Designed for inference / analysis, not training. For training, see Joseph Bloom's SAELens (https://github.com/jbloomAus/SAELens)

    Note that HookedSAETransformer is fairly modular, and doesn't make strong assumptions about the architecture of the SAEs that get attached. We provide HookedSAE as a useful default class, but if you want to eg experiment with other SAE architectures, you can just copy the HookedSAE code into a notebook, edit it, and add instances of the new SAE class to a HookedSAETransformer (e.g. with HookedSAETransformer.add_sae(sae))
    """

    def __init__(self, cfg: Union[HookedSAEConfig, Dict]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedSAEConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedSAEConfig object."
            )
        self.cfg = cfg

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg.d_in, self.cfg.d_sae, dtype=self.cfg.dtype)
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg.d_sae, self.cfg.d_in, dtype=self.cfg.dtype)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(self.cfg.d_sae, dtype=self.cfg.dtype))
        self.b_dec = nn.Parameter(torch.zeros(self.cfg.d_in, dtype=self.cfg.dtype))

        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint()
        self.hook_sae_acts_post = HookPoint()
        self.hook_sae_recons = HookPoint()
        self.hook_sae_error = HookPoint()
        self.hook_sae_output = HookPoint()

        self.to(self.cfg.device)
        self.setup()

    def maybe_reshape_input(
        self,
        input: Float[torch.Tensor, "... d_input"],
        apply_hooks: bool = True,
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Reshape the input to have correct dim.
        No-op for standard SAEs, but useful for hook_z SAEs.
        """
        if apply_hooks:
            self.hook_sae_input(input)

        if input.shape[-1] == self.cfg.d_in:
            x = input
        else:
            # Assume this this is an attention output (hook_z) SAE
            assert self.cfg.hook_name.endswith(
                "_z"
            ), f"You passed in an input shape {input.shape} does not match SAE input size {self.cfg.d_in} for hook_name {self.cfg.hook_name}. This is only supported for attn output (hook_z) SAEs."
            x = einops.rearrange(input, "... n_heads d_head -> ... (n_heads d_head)")
        assert (
            x.shape[-1] == self.cfg.d_in
        ), f"Input shape {x.shape} does not match SAE input size {self.cfg.d_in}"

        return x

    def encode(
        self,
        x: Float[torch.Tensor, "... d_in"],
        apply_hooks: bool = True,
    ) -> Float[torch.Tensor, "... d_sae"]:
        """SAE Encoder.

        Args:
            input: The input tensor of activations to the SAE. Shape [..., d_in].

        Returns:
            output: The encoded output tensor from the SAE. Shape [..., d_sae].
        """
        # Subtract bias term
        x_cent = x - self.b_dec

        # SAE hidden layer pre-RELU  activation
        sae_acts_pre = (
            einops.einsum(x_cent, self.W_enc, "... d_in, d_in d_sae -> ... d_sae")
            + self.b_enc  # [..., d_sae]
        )
        if apply_hooks:
            sae_acts_pre = self.hook_sae_acts_pre(sae_acts_pre)

        # SAE hidden layer post-RELU activation
        sae_acts_post = F.relu(sae_acts_pre)  # [..., d_sae]
        if apply_hooks:
            sae_acts_post = self.hook_sae_acts_post(sae_acts_post)

        return sae_acts_post

    def decode(
        self,
        sae_acts_post: Float[torch.Tensor, "... d_sae"],
        apply_hooks: bool = True,
    ) -> Float[torch.Tensor, "... d_in"]:
        x_reconstruct = (
            einops.einsum(
                sae_acts_post, self.W_dec, "... d_sae, d_sae d_in -> ... d_in"
            )
            + self.b_dec
        )
        if apply_hooks:
            x_reconstruct = self.hook_sae_recons(x_reconstruct)
        return x_reconstruct
        # END WARNING

    def get_error(self, x, apply_hooks=True):
        # Do not hook these as they are only used to compute the error term via a separate path
        sae_acts_post_clean = self.encode(x, apply_hooks=False)
        x_reconstruct_clean = self.decode(sae_acts_post_clean, apply_hooks=False)
        sae_error = x - x_reconstruct_clean
        if apply_hooks:
            sae_error = self.hook_sae_error(sae_error)
        return sae_error

    def maybe_reshape_output(
        self,
        input: Float[torch.Tensor, "... d_input"],
        output: Float[torch.Tensor, "... d_in"],
        apply_hooks: bool = True,
    ):
        output = output.reshape(input.shape)
        if apply_hooks:
            output = self.hook_sae_output(output)
        return output

    def forward(
        self, input: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """SAE Forward Pass.

        Args:
            input: The input tensor of activations to the SAE. Shape [..., d_in].
                Also supports hook_z activations of shape [..., n_heads, d_head], where n_heads * d_head = d_in, for attention output (hook_z) SAEs.

        Returns:
            output: The reconstructed output tensor from the SAE, with the error term optionally added. Same shape as input (eg [..., d_in])
        """
        x = self.maybe_reshape_input(input)
        sae_acts_post = self.encode(x)
        x_reconstruct = self.decode(sae_acts_post)
        if self.cfg.use_error_term:
            sae_error = self.get_error(x)
            x_reconstruct = x_reconstruct + sae_error
        return self.maybe_reshape_output(input, x_reconstruct)
