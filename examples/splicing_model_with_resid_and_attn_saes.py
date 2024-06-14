# pyright: ignore
import transformer_lens as tl
from rich.table import Table
from rich import print as rprint
from circuit_finder.pretrained import load_attn_saes, load_resid_saes
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
    attn_sae.add_hook("hook_sae_acts_post", backward_cache_hook, "bwd")

    resid_saes = load_resid_saes([8])
    resid_sae = resid_saes[8]
    resid_sae.add_hook("hook_sae_acts_post", backward_cache_hook, "bwd")

    # Load model
    model = tl.HookedSAETransformer.from_pretrained("gpt2").cuda()
    answer_tokens = get_answer_tokens(answers, model)  # type: ignore

    # Run model
    with model.saes(saes=[attn_sae, resid_sae]):
        logits, cache = model.run_with_cache(prompts)
        mean_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
        mean_logit_diff.backward()

    # Print the gradients
    for key, value in grad_cache.items():
        rprint(f"{key}: {value.shape}")
