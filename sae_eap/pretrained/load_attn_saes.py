from transformer_lens.utils import download_file_from_hf
from sae_eap.sae.hooked_sae import HookedSAE, HookedSAEConfig
from sae_eap.core.types import LayerIndex
from sae_eap.core.constants import ALL_GPT_2_SMALL_LAYERS
from sae_eap.utils import get_device


def attn_sae_cfg_to_hooked_sae_cfg(attn_sae_cfg):
    new_cfg = {
        "d_sae": attn_sae_cfg["dict_size"],
        "d_in": attn_sae_cfg["act_size"],
        "hook_name": attn_sae_cfg["act_name"],
    }
    return HookedSAEConfig.from_dict(new_cfg)


auto_encoder_runs = [
    "gpt2-small_L0_Hcat_z_lr1.20e-03_l11.80e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L1_Hcat_z_lr1.20e-03_l18.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v5",
    "gpt2-small_L2_Hcat_z_lr1.20e-03_l11.00e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v4",
    "gpt2-small_L3_Hcat_z_lr1.20e-03_l19.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L4_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v7",
    "gpt2-small_L5_Hcat_z_lr1.20e-03_l11.00e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L6_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L7_Hcat_z_lr1.20e-03_l11.10e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L8_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v6",
    "gpt2-small_L9_Hcat_z_lr1.20e-03_l11.20e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L10_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L11_Hcat_z_lr1.20e-03_l13.00e+00_ds24576_bs4096_dc3.16e-06_rsanthropic_rie25000_nr4_v9",
]

hf_repo = "ckkissane/attn-saes-gpt2-small-all-layers"


def parse_layer_for_autoencoder_run(auto_encoder_run: str) -> int:
    return int(auto_encoder_run.split("_")[1][1:])


def load_attn_saes(
    layers: list[int] = ALL_GPT_2_SMALL_LAYERS,
    device: str = get_device(),
    use_error_term: bool = True,
) -> dict[LayerIndex, HookedSAE]:
    """Load the attention-out SAEs trained by Connor Kissane and Rob Kryzgowski

    Reference: https://www.lesswrong.com/posts/DtdzGwFh9dCfsekZZ/sparse-autoencoders-work-on-attention-layer-outputs
    """
    layer_to_sae = {}
    for auto_encoder_run in auto_encoder_runs:
        layer = parse_layer_for_autoencoder_run(auto_encoder_run)
        if layers is not None and layer not in layers:
            continue
        attn_sae_cfg = download_file_from_hf(hf_repo, f"{auto_encoder_run}_cfg.json")
        cfg = attn_sae_cfg_to_hooked_sae_cfg(attn_sae_cfg)
        cfg.use_error_term = use_error_term
        state_dict = download_file_from_hf(
            hf_repo, f"{auto_encoder_run}.pt", force_is_torch=True
        )
        assert state_dict is not None, f"Could not download {auto_encoder_run}.pt"
        hooked_sae = HookedSAE(cfg)
        hooked_sae.load_state_dict(state_dict)  # type: ignore

        layer_to_sae[layer] = hooked_sae.to(device)
    return layer_to_sae
