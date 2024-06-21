from sae_eap.utils import DeviceManager


def test_device_manager_sets_device():
    with DeviceManager.instance().use_device("cpu"):
        assert DeviceManager.instance().get_device() == "cpu"

    with DeviceManager.instance().use_device("cuda"):
        assert DeviceManager.instance().get_device() == "cuda"

    with DeviceManager.instance().use_device("mps"):
        assert DeviceManager.instance().get_device() == "mps"
