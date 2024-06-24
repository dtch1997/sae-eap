import torch
import pickle
import pathlib

from contextlib import contextmanager
from sae_eap.core.patterns import Singleton
from sae_eap.core.types import Device
from transformer_lens import HookedTransformer


@Singleton
class DeviceManager:
    device: Device

    def __init__(self):
        self.device = get_default_device()

    def get_device(self) -> Device:
        return self.device

    def set_device(self, device: Device) -> None:
        self.device = device

    @contextmanager
    def use_device(self, device: Device):
        old_device = self.get_device()
        self.set_device(device)
        yield
        self.set_device(old_device)


def get_default_device() -> Device:
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


def make_save_path(
    savedir: str,
    filename: str,
    *,
    ext: str | None = None,
):
    if "/" in filename:
        raise ValueError("filename should not contain '/'")
    save_dir = pathlib.Path(savedir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    if ext is not None:
        save_path = save_path.with_suffix(ext)
    return save_path


def save_obj_as_pickle(
    obj: object,
    savedir: str,
    filename: str,
):
    """Save an object as a pickle file"""
    save_path = make_save_path(savedir, filename, ext=".pkl")
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)


def load_obj_from_pickle(
    savedir: str,
    filename: str,
) -> object:
    """Load an object from a pickle file"""
    save_path = make_save_path(savedir, filename, ext=".pkl")
    with open(save_path, "rb") as f:
        return pickle.load(f)
