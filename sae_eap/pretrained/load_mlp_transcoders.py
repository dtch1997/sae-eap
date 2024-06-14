from transcoders_slim.transcoder import Transcoder
from transcoders_slim.load_pretrained import load_pretrained
from transcoders_slim.sae_training.config import LanguageModelSAERunnerConfig

from sae_eap.core.types import LayerIndex
from sae_eap.core.constants import ALL_GPT_2_SMALL_LAYERS
from sae_eap.sae.hooked_transcoder import (
    HookedTranscoder,
    HookedTranscoderConfig,
)
from sae_eap.utils import get_device


def get_filenames(layers: list[int]) -> list[str]:
    return [
        f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.ln2.hook_normalized_24576.pt"
        for layer in layers
    ]


def parse_layer_of_module_name(module_name: str) -> LayerIndex:
    return int(module_name.split(".")[1])


def ts_tc_cfg_to_hooked_tc_cfg(
    tc_cfg: LanguageModelSAERunnerConfig,
) -> HookedTranscoderConfig:
    new_cfg = {
        "d_sae": tc_cfg.d_sae,
        "d_in": tc_cfg.d_in,
        "d_out": tc_cfg.d_out,
        "hook_name": tc_cfg.hook_point,
        "hook_name_out": tc_cfg.out_hook_point,
    }
    return HookedTranscoderConfig.from_dict(new_cfg)


def ts_tc_to_hooked_tc(
    sl_sae: Transcoder,
) -> HookedTranscoder:
    state_dict = sl_sae.state_dict()
    cfg = ts_tc_cfg_to_hooked_tc_cfg(sl_sae.cfg)
    tl_sae = HookedTranscoder(cfg)
    tl_sae.load_state_dict(state_dict)
    return tl_sae


def load_hooked_mlp_transcoders(
    layers: list[int] = ALL_GPT_2_SMALL_LAYERS,
    device: str = get_device(),
    use_error_term: bool = True,
) -> dict[LayerIndex, HookedTranscoder]:
    transcoders_dict = load_pretrained(get_filenames(layers))
    hooked_transcoders = {}
    for module_name, transcoder in transcoders_dict.items():
        layer = parse_layer_of_module_name(module_name)
        hooked_transcoder = ts_tc_to_hooked_tc(transcoder).to(device)
        hooked_transcoder.cfg.use_error_term = use_error_term
        hooked_transcoders[layer] = hooked_transcoder

    return hooked_transcoders
