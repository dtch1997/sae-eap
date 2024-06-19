import torch
from sae_eap.core.types import Device
from transformer_lens import HookedTransformer


def get_device() -> Device:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Parse the PyTorch version to check if it's below version 2.0
        major_version = int(torch.__version__.split(".")[0])
        if major_version >= 2:
            return "mps"
    else:
        return "cpu"

    raise RuntimeError("Should not reach here!")


def get_npos_and_input_lengths(model: HookedTransformer, inputs: list[str]):
    assert model.tokenizer is not None
    tokenized = model.tokenizer(
        inputs, padding="longest", return_tensors="pt", add_special_tokens=True
    )
    # assert tokenized is not None
    # assert tokenized.attention_mask is not None
    n_pos = 1 + tokenized.attention_mask.size(1)
    input_lengths = 1 + tokenized.attention_mask.sum(1)
    return n_pos, input_lengths
