import pytest

from sae_eap.core.types import Model
from sae_eap.data.handler import SinglePromptHandler

# Rough estimate of logit diff between John and Mary
IOI_LOGIT_DIFF = 3


@pytest.fixture
def single_prompt_handler(gpt2_small_model) -> SinglePromptHandler:
    return SinglePromptHandler(
        model=gpt2_small_model,
        clean_prompt="When John and Mary went to the shops, John gave a bag to",
        corrupt_prompt="When John and Mary went to the shops, Mary gave a bag to",
        answer=" Mary",
        wrong_answer=" John",
    )


def test_gpt_2_small_ioi_single_prompt(
    gpt2_small_model: Model, single_prompt_handler: SinglePromptHandler
):
    # Test the function with a simple example
    logits = single_prompt_handler.get_logits(gpt2_small_model, input="clean")
    metric = single_prompt_handler.get_metric(logits)
    assert metric > IOI_LOGIT_DIFF

    corrupt_logits = single_prompt_handler.get_logits(gpt2_small_model, input="corrupt")
    corrupt_metric = single_prompt_handler.get_metric(corrupt_logits)
    assert corrupt_metric < -IOI_LOGIT_DIFF
