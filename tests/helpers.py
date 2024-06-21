import copy

from transformer_lens import HookedTransformer
from sae_eap.utils import DeviceManager
from sae_eap.model.load_pretrained import load_model

DeviceManager.instance().use_device("cpu")

SOLU_1L_MODEL = "solu-1l"
TINYSTORIES_MODEL = "tiny-stories-1M"
GPT2_SMALL_MODEL = "gpt2-small"


MODEL_CACHE: dict[str, HookedTransformer] = {}


def load_model_cached(model_name: str) -> HookedTransformer:
    """
    helper to avoid unnecessarily loading the same model multiple times.
    NOTE: if the model gets modified in tests this will not work.
    """
    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = load_model(model_name)
    # we copy here to prevent sharing state across tests
    return copy.deepcopy(MODEL_CACHE[model_name])
