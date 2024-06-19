import pytest

from tests.helpers import load_model_cached, TINYSTORIES_MODEL, GPT2_SMALL_MODEL


@pytest.fixture
def ts_model():
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture
def gpt2_small_model():
    return load_model_cached(GPT2_SMALL_MODEL)
