import pytest
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from tests.helpers import TINYSTORIES_MODEL, build_sae_cfg, load_model_cached
from typing import cast


@pytest.fixture()
def model(device: str) -> HookedTransformer:
    return cast(HookedTransformer, load_model_cached(TINYSTORIES_MODEL).to(device))


@pytest.fixture()
def sae(device) -> SAE:
    return SAE(build_sae_cfg()).to(device)


# @pytest.fixture()
# def sae_dict(sae: SparseAutoencoder) -> dict[ModuleName, SparseAutoencoder]:
#     sae_dict = {ModuleName(sae.cfg.hook_point): sae}
#     return sae_dict


@pytest.fixture()
def device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


@pytest.fixture()
def text() -> str:
    return "Hello world"
