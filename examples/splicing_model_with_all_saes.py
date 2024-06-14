# pyright: ignore
import transformer_lens as tl
from rich.table import Table
from rich import print as rprint
from circuit_finder.pretrained import (
    load_attn_saes,
    load_resid_saes,
    load_hooked_mlp_transcoders,
)
from circuit_finder.core.hooked_transcoder import HookedTranscoderReplacementContext
from circuit_finder.utils import get_answer_tokens, logits_to_ave_logit_diff
from circuit_finder.data import ioi


if __name__ == "__main__":
    # Load the data
    prompts, _, answers = ioi.get_ioi_data()

    # Print the data
    table = Table("Prompt", "Correct", "Incorrect", title="Prompts & Answers:")
    for prompt, answer in zip(prompts, answers):
        table.add_row(prompt, repr(answer[0]), repr(answer[1]))
    rprint(table)

    # Initialize the cache to store gradients
    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    # Initialize SAEs
    attn_saes = load_attn_saes([8])
    attn_sae = attn_saes[8]
    attn_sae.cfg.use_error_term = True
    attn_sae.add_hook("hook_sae_acts_post", backward_cache_hook, "bwd")

    resid_saes = load_resid_saes([8])
    resid_sae = resid_saes[8]
    resid_sae.cfg.use_error_term = True
    resid_sae.add_hook("hook_sae_acts_post", backward_cache_hook, "bwd")

    mlp_transcoders = load_hooked_mlp_transcoders([8])
    mlp_transcoder = mlp_transcoders[8]
    # TODO: For some reason, this doesn't work?
    mlp_transcoder.cfg.use_error_term = True

    # NOTE: these will have name 'blocks.8.mlp.transcoder.hook_sae_XXX'
    mlp_transcoder.add_hook("hook_sae_input", backward_cache_hook, "bwd")
    mlp_transcoder.add_hook("hook_sae_acts_pre", backward_cache_hook, "bwd")
    mlp_transcoder.add_hook("hook_sae_acts_post", backward_cache_hook, "bwd")
    mlp_transcoder.add_hook("hook_sae_recons", backward_cache_hook, "bwd")

    # Load model
    model = tl.HookedSAETransformer.from_pretrained("gpt2").cuda()
    answer_tokens = get_answer_tokens(answers, model)  # type: ignore

    # Run model
    with HookedTranscoderReplacementContext(
        model,  # type: ignore
        transcoders=[mlp_transcoder],
    ) as context:
        for wrapped_transcoder in context.wrapped_transcoders:
            # NOTE: these will have name 'blocks.8.mlp.hook_sae_XXX'
            wrapped_transcoder.add_hook("hook_sae_error", backward_cache_hook, "bwd")
            wrapped_transcoder.add_hook("hook_sae_output", backward_cache_hook, "bwd")
        with model.saes(saes=[attn_sae, resid_sae]):
            logits, cache = model.run_with_cache(prompts)
            mean_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
            mean_logit_diff.backward()

    # Print the activtions
    print()
    print("Cached activations: ")
    for key, value in cache.items():
        if "hook_sae" in key:
            rprint(f"{key}: {value.shape}")
    # Print the gradients
    print()
    print("Cached gradients: ")
    for key, value in grad_cache.items():
        rprint(f"{key}: {value.shape}")
