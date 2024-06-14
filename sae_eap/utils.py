import torch
from sae_eap.core.types import Device


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
