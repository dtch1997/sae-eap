import pathlib

GPT_2_SMALL_MODEL_NAME: str = "gpt2"
GPT_2_SMALL_ALL_LAYERS: list[int] = list(range(12))

ProjectDir = pathlib.Path(__file__).parent.parent
