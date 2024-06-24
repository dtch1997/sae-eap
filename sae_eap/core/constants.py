import pathlib
from transformer_lens import HookedTransformerConfig
from transformer_lens.loading_from_pretrained import get_pretrained_model_config

ProjectDir = pathlib.Path(__file__).parent.parent.parent

GPT_2_SMALL_MODEL_NAME: str = "gpt2"
GPT_2_SMALL_MODEL_CONFIG: HookedTransformerConfig = get_pretrained_model_config(
    GPT_2_SMALL_MODEL_NAME
)
GPT_2_SMALL_ALL_LAYERS: list[int] = list(range(12))
