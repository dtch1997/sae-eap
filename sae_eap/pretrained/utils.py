import torch
from sae_lens import SparseAutoencoder
from sae_lens.training.train_sae_on_language_model import LanguageModelSAERunnerConfig

from circuit_finder.core.hooked_sae import HookedSAE
from circuit_finder.core.hooked_sae_config import HookedSAEConfig


def sl_sae_cfg_to_tl_sae_cfg(
    resid_sae_cfg: LanguageModelSAERunnerConfig,
) -> HookedSAEConfig:
    new_cfg = {
        "d_sae": resid_sae_cfg.d_sae,
        "d_in": resid_sae_cfg.d_in,
        "hook_name": resid_sae_cfg.hook_point,
    }
    return HookedSAEConfig.from_dict(new_cfg)


def sl_sae_to_tl_sae(
    sl_sae: SparseAutoencoder,
) -> HookedSAE:
    state_dict = sl_sae.state_dict()
    # NOTE: sae-lens uses a 'scaling factor'
    # For now, just check this is 1 and then remove it
    torch.allclose(
        state_dict["scaling_factor"], torch.ones_like(state_dict["scaling_factor"])
    )
    state_dict.pop("scaling_factor")

    cfg = sl_sae_cfg_to_tl_sae_cfg(sl_sae.cfg)
    tl_sae = HookedSAE(cfg)
    tl_sae.load_state_dict(state_dict)
    return tl_sae
