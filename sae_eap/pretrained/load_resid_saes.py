"""Load Joseph Bloom's residual SAEs"""

from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

from sae_eap.sae.hooked_sae import HookedSAE
from sae_eap.core.types import LayerIndex
from sae_eap.core.constants import ALL_GPT_2_SMALL_LAYERS
from sae_eap.pretrained.utils import sl_sae_to_tl_sae
from sae_eap.utils import get_device


def get_hook_point_from_layer(layer: LayerIndex) -> str:
    return f"blocks.{layer}.hook_resid_pre"


def load_resid_saes(
    layers: list[int] = ALL_GPT_2_SMALL_LAYERS + [12],
    device: str = get_device(),
) -> dict[LayerIndex, HookedSAE]:
    layer_to_sae = {}
    for layer in layers:
        if 0 <= layer < 12:
            hook_point = f"blocks.{layer}.hook_resid_pre"
        elif layer == 12:
            hook_point = "blocks.11.hook_resid_post"
        else:
            raise ValueError(f"Invalid layer: {layer}")

        # Load the SAE
        (sae_dict, _) = get_gpt2_res_jb_saes(hook_point)
        sae = sae_dict[hook_point]

        # convert to HookedSAE
        hooked_sae = sl_sae_to_tl_sae(sae)
        layer_to_sae[layer] = hooked_sae.to(device)
    return layer_to_sae
