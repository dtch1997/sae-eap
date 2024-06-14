import pytest
from circuit_finder.pretrained import load_attn_saes


# TODO: test for all layers
@pytest.mark.parametrize("layer", [11])
def test_load_attn_saes(layer: int):
    sae = load_attn_saes(layers=[layer])
    assert sae[layer] is not None
