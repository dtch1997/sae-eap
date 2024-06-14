import pytest
from circuit_finder.pretrained import load_resid_saes


# TODO: test for all layers
@pytest.mark.parametrize("layer", [12])
def test_load_attn_saes(layer: int):
    sae_dict = load_resid_saes(layers=[layer])
    assert sae_dict[layer] is not None
