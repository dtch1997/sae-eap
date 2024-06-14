import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_lens as tl
from jaxtyping import Float

from typing import Dict, Union, Sequence
from transformer_lens.hook_points import HookPoint, HookedRootModule

from circuit_finder.core.types import LayerIndex
from circuit_finder.core.hooked_transcoder_config import HookedTranscoderConfig


class HookedTranscoder(HookedRootModule):
    """Hooked Transcoder"""

    def __init__(self, cfg: Union[HookedTranscoderConfig, Dict]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTranscoderConfig(**cfg)
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
        # NOTE: There are two bias terms here with different roles.
        # The b_dec term is subtracted from x at the start of the forward pass.
        # The b_dec_out term is added in the decoding step.
        self.b_dec = nn.Parameter(torch.zeros(self.cfg.d_in, dtype=self.cfg.dtype))
        self.b_dec_out = nn.Parameter(torch.zeros(self.cfg.d_in, dtype=self.cfg.dtype))

        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint()
        self.hook_sae_acts_post = HookPoint()
        self.hook_sae_recons = HookPoint()

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
        # NOTE: here we use b_dec!
        # Reference: https://github.com/jacobdunefsky/transcoder_circuits/blob/7b44d870a5a301ef29eddfd77cb1f4dca854760a/sae_training/sparse_autoencoder.py#L110C1-L112C50
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
            # NOTE: Here we use b_dec_out!
            # Reference: https://github.com/jacobdunefsky/transcoder_circuits/blob/7b44d870a5a301ef29eddfd77cb1f4dca854760a/sae_training/sparse_autoencoder.py#L124C1-L133C33
            + self.b_dec_out
        )
        if apply_hooks:
            x_reconstruct = self.hook_sae_recons(x_reconstruct)
        return x_reconstruct

    def maybe_reshape_output(
        self,
        input: Float[torch.Tensor, "... d_input"],
        output: Float[torch.Tensor, "... d_in"],
    ):
        output = output.reshape(input.shape)
        return output

    def forward(
        self, input: Float[torch.Tensor, "... d_in"], apply_hooks: bool = True
    ) -> Float[torch.Tensor, "... d_in"]:
        """SAE Forward Pass.

        Args:
            input: The input tensor of activations to the SAE. Shape [..., d_in].
                Also supports hook_z activations of shape [..., n_heads, d_head], where n_heads * d_head = d_in, for attention output (hook_z) SAEs.

        Returns:
            output: The reconstructed output tensor from the SAE, with the error term optionally added. Same shape as input (eg [..., d_in])
        """
        output, _ = self.get_recons_and_act_post(input, apply_hooks=apply_hooks)
        return output

    def get_recons_and_act_post(
        self, input: Float[torch.Tensor, "... d_in"], apply_hooks: bool = True
    ) -> tuple[Float[torch.Tensor, "... d_in"], Float[torch.Tensor, "... d_model"]]:
        x = self.maybe_reshape_input(input, apply_hooks=apply_hooks)
        sae_acts_post = self.encode(x, apply_hooks=apply_hooks)
        x_reconstruct = self.decode(sae_acts_post, apply_hooks=apply_hooks)
        output = self.maybe_reshape_output(input, x_reconstruct)
        return output, sae_acts_post


class HookedTranscoderWrapper(HookedRootModule):
    """Wrapper around transcoder and the MLP it replaces"""

    def __init__(
        self,
        transcoder: HookedTranscoder,
        mlp: nn.Module,
    ):
        super().__init__()
        self.transcoder = transcoder

        self.hook_sae_error = HookPoint()
        self.hook_sae_output = HookPoint()
        self.setup()

        # NOTE: we want to exclude the MLP from the HookedTransformer's hook points.
        # So we add mlp after setup.
        # Suggested by Arthur
        self.mlp = mlp
        self.mlp.to(transcoder.cfg.device)

    @property
    def cfg(self):
        return self.transcoder.cfg

    @property
    def W_dec(self):
        return self.transcoder.W_dec

    @property
    def b_dec(self):
        return self.transcoder.b_dec

    @property
    def b_dec_out(self):
        return self.transcoder.b_dec_out

    @property
    def W_enc(self):
        return self.transcoder.W_enc

    @property
    def b_enc(self):
        return self.transcoder.b_enc

    def forward(self, x, apply_hooks=True):
        sae_output = self.transcoder(x)
        if not self.cfg.use_error_term:
            # Not using error term
            if apply_hooks:
                sae_output = self.hook_sae_output(sae_output)
            return sae_output

        else:
            # Using error term
            sae_out_clean = self.transcoder(x, apply_hooks=False)
            mlp_out = self.mlp(x)
            sae_error = mlp_out - sae_out_clean

            if apply_hooks:
                sae_error = self.hook_sae_error(sae_error)
            if apply_hooks:
                sae_output = self.hook_sae_output(sae_output + sae_error)
            return sae_output


def get_layer_of_hook_name(hook_point):
    return int(hook_point.split(".")[1])


class HookedTranscoderReplacementContext:
    """Context manager to replace MLP sublayers with transcoders"""

    model: tl.HookedTransformer
    transcoders: Sequence[HookedTranscoder]
    layers: Sequence[LayerIndex]
    original_mlps: dict[LayerIndex, nn.Module]

    def __init__(
        self, model: tl.HookedTransformer, transcoders: Sequence[HookedTranscoder]
    ):
        self.layers = [get_layer_of_hook_name(t.cfg.hook_name) for t in transcoders]
        self.original_mlps = {layer: model.blocks[layer].mlp for layer in self.layers}
        self.transcoders = transcoders
        self.wrapped_transcoders = []
        self.model = model

    def __enter__(self):
        # Replace all MLPs with transcoder
        for layer, transcoder in zip(self.layers, self.transcoders):
            mlp = self.model.blocks[layer].mlp
            wrapped_transcoder = HookedTranscoderWrapper(transcoder, mlp)
            self.wrapped_transcoders.append(wrapped_transcoder)
            self.model.blocks[layer].mlp = wrapped_transcoder

        self.model.setup()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # Restore original MLPs
        for layer, mlp in self.original_mlps.items():
            self.model.blocks[layer].mlp = mlp

        self.model.setup()
