from circuit_finder.core.hooked_transcoder import (
    HookedTranscoder,
    HookedTranscoderWrapper,
    HookedTranscoderReplacementContext,
)
from circuit_finder.core.hooked_transcoder_config import HookedTranscoderConfig
from circuit_finder.pretrained.load_mlp_transcoders import (
    load_mlp_transcoders,
    ts_tc_to_hooked_tc,
)


import pytest
import torch

from transformer_lens import HookedSAETransformer
from circuit_finder.constants import device

MODEL = "solu-1l"
prompt = "Hello World!"


class Counter:
    def __init__(self):
        self.count = 0

    def inc(self, *args, **kwargs):
        self.count += 1


@pytest.fixture(scope="module")
def model():
    model = HookedSAETransformer.from_pretrained(MODEL)
    yield model
    model.reset_saes()


def get_site(act_name):
    site = act_name.split(".")[2:]
    site = ".".join(site)
    return site


def get_transcoder_config(model, act_name_in, act_name_out):
    site_to_size = {
        "ln2.hook_normalized": model.cfg.d_model,
        "hook_mlp_out": model.cfg.d_model,
    }
    site_in = get_site(act_name_in)
    assert site_in == "ln2.hook_normalized"
    d_in = site_to_size[site_in]

    site_out = get_site(act_name_out)
    d_out = site_to_size[site_out]

    return HookedTranscoderConfig(
        d_in=d_in,
        d_out=d_out,
        d_sae=d_in * 2,
        hook_name=act_name_in,
        hook_name_out=act_name_out,
    )


def test_forward_reconstructs_input(model):
    """Verfiy that the HookedSAE returns an output with the same shape as the input activations."""
    act_in = "blocks.0.ln2.hook_normalized"
    act_out = "blocks.0.hook_mlp_out"
    tc_cfg = get_transcoder_config(
        model,
        act_in,
        act_out,
    )
    hooked_tc = HookedTranscoder(tc_cfg)

    _, cache = model.run_with_cache(prompt)
    x_in = cache[act_in]
    x_out = cache[act_out]

    sae_output = hooked_tc(x_in)
    assert sae_output.shape == x_out.shape


def test_run_with_cache(model):
    """Verifies that run_with_cache caches SAE activations"""
    act_in = "blocks.0.ln2.hook_normalized"
    act_out = "blocks.0.hook_mlp_out"
    tc_cfg = get_transcoder_config(
        model,
        act_in,
        act_out,
    )
    tc_cfg.use_error_term = True
    hooked_tc = HookedTranscoder(tc_cfg)

    expected_hook_names = [
        "transcoder.hook_sae_input",
        "transcoder.hook_sae_recons",
        "transcoder.hook_sae_acts_pre",
        "transcoder.hook_sae_acts_post",
        "hook_sae_error",
        "hook_sae_output",
    ]

    with HookedTranscoderReplacementContext(model, [hooked_tc]):
        _, cache = model.run_with_cache(prompt)

    for hook_name in expected_hook_names:
        assert "blocks.0.mlp." + hook_name in cache


def test_error_term(model):
    """Verifies that that if we use error_terms, HookedSAE returns an output that is equal to the input activations."""
    act_in = "blocks.0.ln2.hook_normalized"
    act_out = "blocks.0.hook_mlp_out"
    tc_cfg = get_transcoder_config(
        model,
        act_in,
        act_out,
    )
    tc_cfg.use_error_term = True
    hooked_tc = HookedTranscoder(tc_cfg)

    _, orig_cache = model.run_with_cache(prompt)

    # Test hooked transcoder wrapper
    wrapped_hook_tc = HookedTranscoderWrapper(hooked_tc, model.blocks[0].mlp)

    in_orig = orig_cache[act_in]
    out_orig = orig_cache[act_out]
    sae_out = wrapped_hook_tc(in_orig)
    assert sae_out.shape == out_orig.shape
    assert torch.allclose(sae_out, out_orig, atol=1e-6)

    # Test replacement context
    with HookedTranscoderReplacementContext(model, [hooked_tc]):
        _, spliced_cache = model.run_with_cache(prompt)

    out_orig = orig_cache[act_out]
    out_spliced = spliced_cache[act_out]

    assert out_orig.shape == out_spliced.shape
    assert torch.allclose(out_orig, out_spliced, atol=1e-6)


def test_error_grads(model):
    """Verifies that if we use error terms, the error term has a gradient"""
    act_in = "blocks.0.ln2.hook_normalized"
    act_out = "blocks.0.hook_mlp_out"
    tc_cfg = get_transcoder_config(
        model,
        act_in,
        act_out,
    )
    tc_cfg.use_error_term = True
    hooked_tc = HookedTranscoder(tc_cfg)
    wrapped_hook_tc = HookedTranscoderWrapper(hooked_tc, model.blocks[0].mlp)

    # Define SAE input
    _, orig_cache = model.run_with_cache(prompt)
    x = orig_cache[act_in]
    x_out = orig_cache[act_out]

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    wrapped_hook_tc.add_hook("hook_sae_error", backward_cache_hook, "bwd")

    sae_output = wrapped_hook_tc(x)
    assert sae_output.shape == x_out.shape
    assert torch.allclose(sae_output, x_out, atol=1e-6)

    value = sae_output.sum()
    value.backward()
    wrapped_hook_tc.reset_hooks()

    assert len(grad_cache) == 1
    assert "hook_sae_error" in grad_cache

    # NOTE: The output is linear in the error, hence analytic gradient is one
    grad = grad_cache["hook_sae_error"]
    analytic_grad = torch.ones_like(grad)
    assert torch.allclose(grad, analytic_grad, atol=1e-6)


def test_feature_grads_with_error_term(model):
    """Verifies that pytorch backward computes the correct feature gradients when using error_terms. Motivated by the need to compute feature gradients for attribution patching."""

    # Load Transcoder
    act_in = "blocks.0.ln2.hook_normalized"
    act_out = "blocks.0.hook_mlp_out"
    tc_cfg = get_transcoder_config(
        model,
        act_in,
        act_out,
    )
    tc_cfg.use_error_term = True
    hooked_tc = HookedTranscoder(tc_cfg)
    wrapped_hook_tc = HookedTranscoderWrapper(hooked_tc, model.blocks[0].mlp)

    # Define SAE input
    _, orig_cache = model.run_with_cache(prompt)
    x = orig_cache[act_in]
    x_out = orig_cache[act_out]

    # Cache gradients with respect to feature acts
    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    wrapped_hook_tc.add_hook(
        "transcoder.hook_sae_acts_post", backward_cache_hook, "bwd"
    )
    wrapped_hook_tc.add_hook("hook_sae_output", backward_cache_hook, "bwd")

    sae_output = wrapped_hook_tc(x)
    assert torch.allclose(sae_output, x_out, atol=1e-6)
    value = sae_output.sum()
    value.backward()
    wrapped_hook_tc.reset_hooks()

    # Compute gradient analytically
    analytic_grad = grad_cache["hook_sae_output"] @ wrapped_hook_tc.W_dec.T

    # Compare analytic gradient with pytorch computed gradient
    assert torch.allclose(
        grad_cache["transcoder.hook_sae_acts_post"], analytic_grad, atol=1e-6
    )


def test_hooked_transcoder_forward_equals_transcoder_forward():
    tc = load_mlp_transcoders([0])[0]
    hooked_tc = ts_tc_to_hooked_tc(tc)

    x = torch.randn(1, 4, 768, device=device)
    tc_out = tc(x)[0]
    hooked_tc_out = hooked_tc(x)
    assert torch.allclose(tc_out, hooked_tc_out, atol=1e-6)
