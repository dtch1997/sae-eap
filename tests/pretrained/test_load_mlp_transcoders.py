from circuit_finder.pretrained import load_mlp_transcoders, load_hooked_mlp_transcoders
from circuit_finder.core.hooked_transcoder import HookedTranscoder


def test_load_mlp_transcoders():
    transcoder = load_mlp_transcoders(layers=[0])
    assert transcoder[0] is not None


def test_load_hooked_transcoders():
    transcoder = load_hooked_mlp_transcoders(layers=[0])
    assert transcoder[0] is not None
    assert isinstance(transcoder[0], HookedTranscoder)
