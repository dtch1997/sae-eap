# flake8: noqa
# %%
"""Demonstrate how to do linear edge attribution patching using LEAP"""

# Imports and downloads
import sys

sys.path.append("/root/circuit-finder")
print(sys.path)

import transformer_lens as tl
from torch import Tensor
from jaxtyping import Int
from typing import Callable
from circuit_finder.patching.eap_graph import EAPGraph
from circuit_finder.plotting import show_attrib_graph
import torch
import gc
from circuit_finder.patching.leap import last_token_logit
from tqdm import tqdm

from circuit_finder.pretrained import (
    load_attn_saes,
    load_mlp_transcoders,
)
from circuit_finder.patching.leap import (
    preprocess_attn_saes,
    LEAP,
    LEAPConfig,
)

from circuit_finder.patching.patched_fp import patched_fp

# Load models
model = tl.HookedTransformer.from_pretrained(
    "gpt2",
    device="cuda",
    fold_ln=True,
    center_writing_weights=True,
    center_unembed=True,
)

attn_saes = load_attn_saes()
attn_saes = preprocess_attn_saes(attn_saes, model)  # type: ignore
transcoders = load_mlp_transcoders()

# %%
# Define dataset
tokens = model.to_tokens(
    [
        "When John and Mary were at the store, John gave a bottle to Mary",
        "When Linda and Tom were in the park, Linda threw a ball to Tom",
        "While Sarah and Jamie are on the run, Jamie gives a gun to Sarah",
        "When Hugh and Susan came to the party, Susan passed a drink to Hugh",
        "Since Tom and Jim are best of friends, Tom gives a hug to Jim",
    ]
)
corrupt_tokens = model.to_tokens(
    [
        "When Alice and Bob were at the store, Charlie gave a bottle to Dan",
        "When Alice and Bob were in the park, Charlie threw a ball to Dan",
        "When Alice and Bob are on the run, Charlie gives a gun to Dan",
        "When Alice and Bob came to the party, Charlie passed a drink to Dan",
        "Since Alice and Bob are best of friends, Charlie gives a hug to Dan",
    ]
)
# %%
# Specify what to do with error nodes when calculating faithfulness curves. Options are:
# ablate_errors = False  ->  we don't ablate error nodes
# ablate_errors = "bm"   ->  we batchmean-ablate error nodes
# ablate_errors = "zero" ->  we zero-ablate error nodes (warning: this gives v bad performance)
ablate_errors = False

# to get ceiling, we run clean model
model.reset_hooks()
ceiling = last_token_logit(model, tokens).item()

# to get floor, use empty graph, i.e. ablate everything
graph = EAPGraph([])
floor = patched_fp(
    model,
    graph,
    tokens,
    last_token_logit,
    transcoders,
    attn_saes,
    ablate_errors=ablate_errors,  # if bm, error nodes are mean-ablated
    first_ablated_layer=2,  # Marks et al don't ablate first 2 layers
).item()
torch.cuda.empty_cache()
gc.collect()

# now sweep over thresholds to get graphs with variety of numbers of nodes
# for each graph we calculate faithfulness
num_nodes_list = []
metrics_list = []
for threshold in tqdm(
    [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.03, 0.06, 0.1]
):
    model.reset_hooks()
    cfg = LEAPConfig(threshold=threshold, contrast_pairs=False, chained_attribs=True)
    leap = LEAP(
        cfg, tokens, model, attn_saes, transcoders, corrupt_tokens=corrupt_tokens
    )
    leap.get_graph(verbose=False)
    graph = EAPGraph(leap.graph)

    num_nodes_list.append(len(graph.get_src_nodes()))
    # show_attrib_graph(graph)  # rendering the graphs slows things down a lot

    del leap
    gc.collect()
    torch.cuda.empty_cache()

    metric = patched_fp(
        model,
        graph,
        tokens,
        last_token_logit,
        transcoders,
        attn_saes,
        ablate_errors=ablate_errors,  # if bm, error nodes are mean-ablated
        first_ablated_layer=2,  # Marks et al don't ablate first 2 layers
    )

    torch.cuda.empty_cache()
    gc.collect()
    metrics_list.append(metric.item())

faith = [(metric - floor) / (ceiling - floor) for metric in metrics_list]

# %%
# plot the faithfulness curve
import plotly.express as px

fig = px.line(
    x=num_nodes_list,
    y=faith,
    labels={"x": "Number of Nodes", "y": "Faithfulness"},
    title="IOI - error nodes not ablated",
)
fig.show()
