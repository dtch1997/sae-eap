from sae_eap.utils import DeviceManager
from transformer_lens import HookedTransformer


def load_model(model_name: str) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(
        model_name,
        device=DeviceManager.instance().get_device(),
    )
    model.set_use_attn_result(True)
    return model
