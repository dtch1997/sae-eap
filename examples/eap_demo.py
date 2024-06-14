# type: ignore
# flake8: noqa
# %%
import torch

import torch as t
from torch import Tensor
import einops

from transformer_lens import HookedTransformer

from eap.eap_wrapper import EAP

from jaxtyping import Float

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
print(f"Device: {device}")

# %%
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)

# %%
from notebooks.ioi_dataset import IOIDataset, format_prompt, make_table

N = 25
clean_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=device,
)
corr_dataset = clean_dataset.gen_flipped_prompts("ABC->XYZ, BAB->XYZ")

make_table(
    colnames=["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
    cols=[
        map(format_prompt, clean_dataset.sentences),
        model.to_string(clean_dataset.s_tokenIDs).split(),
        model.to_string(clean_dataset.io_tokenIDs).split(),
        map(format_prompt, clean_dataset.sentences),
    ],
    title="Sentences from IOI vs ABC distribution",
)


# %%
def ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    ioi_dataset: IOIDataset,
    per_prompt: bool = False,
):
    """
    Return average logit difference between correct and incorrect answers
    """
    # Get logits for indirect objects
    batch_size = logits.size(0)
    io_logits = logits[
        range(batch_size),
        ioi_dataset.word_idx["end"][:batch_size],
        ioi_dataset.io_tokenIDs[:batch_size],
    ]
    s_logits = logits[
        range(batch_size),
        ioi_dataset.word_idx["end"][:batch_size],
        ioi_dataset.s_tokenIDs[:batch_size],
    ]
    # Get logits for subject
    logit_diff = io_logits - s_logits
    return logit_diff if per_prompt else logit_diff.mean()


with t.no_grad():
    clean_logits = model(clean_dataset.toks)
    corrupt_logits = model(corr_dataset.toks)
    clean_logit_diff = ave_logit_diff(clean_logits, clean_dataset).item()
    corrupt_logit_diff = ave_logit_diff(corrupt_logits, corr_dataset).item()


def ioi_metric(
    logits: Float[Tensor, "batch seq_len d_vocab"],
    corrupted_logit_diff: float = corrupt_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
    ioi_dataset: IOIDataset = clean_dataset,
):
    patched_logit_diff = ave_logit_diff(logits, ioi_dataset)
    return (patched_logit_diff - corrupted_logit_diff) / (
        clean_logit_diff - corrupted_logit_diff
    )


def negative_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    return -ioi_metric(logits)


# Get clean and corrupt logit differences
with t.no_grad():
    clean_metric = ioi_metric(
        clean_logits, corrupt_logit_diff, clean_logit_diff, clean_dataset
    )
    corrupt_metric = ioi_metric(
        corrupt_logits, corrupt_logit_diff, clean_logit_diff, corr_dataset
    )

print(f"Clean direction: {clean_logit_diff}, Corrupt direction: {corrupt_logit_diff}")
print(f"Clean metric: {clean_metric}, Corrupt metric: {corrupt_metric}")

# %%
model.reset_hooks()

graph = EAP(
    model,
    clean_dataset.toks,
    corr_dataset.toks,
    ioi_metric,
    upstream_nodes=["mlp", "head"],
    downstream_nodes=["mlp", "head"],
    batch_size=25,
)

top_edges = graph.top_edges(n=10, abs_scores=True)
for from_edge, to_edge, score in top_edges:
    print(f"{from_edge} -> [{round(score, 3)}] -> {to_edge}")

# %%
