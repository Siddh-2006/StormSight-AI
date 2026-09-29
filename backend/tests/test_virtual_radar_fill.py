import torch

from app.ml.virtual_radar_fill import VRF_INPUT_CHANNELS, VirtualRadarFillUNet


def test_output_shape_and_range():
    model = VirtualRadarFillUNet()
    x = torch.rand(2, len(VRF_INPUT_CHANNELS), 64, 64)
    out = model(x)
    assert out.shape == (2, 2, 64, 64)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)  # sigmoided


def test_split_output():
    model = VirtualRadarFillUNet()
    x = torch.rand(1, len(VRF_INPUT_CHANNELS), 32, 32)
    raw = model(x)
    radar, confidence = VirtualRadarFillUNet.split_output(raw)
    assert radar.shape == (1, 32, 32)
    assert confidence.shape == (1, 32, 32)


def test_radar_dbz_excluded_from_inputs():
    # The whole point of this model is to reconstruct radar_dbz from
    # everything ELSE — it must never appear in its own input list.
    assert "radar_dbz" not in VRF_INPUT_CHANNELS


def test_gradients_flow():
    model = VirtualRadarFillUNet()
    x = torch.rand(1, len(VRF_INPUT_CHANNELS), 32, 32, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None
    assert not torch.all(x.grad == 0)
