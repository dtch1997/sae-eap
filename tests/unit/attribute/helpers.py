from sae_eap.core.types import Model
from sae_eap.data.handler import SinglePromptHandler


def make_single_prompt_handler(model: Model) -> SinglePromptHandler:
    return SinglePromptHandler(
        model=model,
        clean_prompt="clean",
        corrupt_prompt="dirty",
        answer=" answer",
        wrong_answer=" wrong",
    )
