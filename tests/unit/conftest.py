import pytest

from tests.unit.helpers import load_model_cached, TINYSTORIES_MODEL


@pytest.fixture
def ts_model():
    return load_model_cached(TINYSTORIES_MODEL)
