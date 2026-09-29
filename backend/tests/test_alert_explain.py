import numpy as np

from app.ml.alert_explain import explain_cell


def test_explain_cell_names_rising_channel():
    # 2 time steps, 5 channels (CHANNELS order), 1x1 spatial cell.
    # Channel 2 (lightning_density) rises sharply; everything else flat.
    history = np.zeros((2, 5, 1, 1), dtype=np.float32)
    history[1, 2, 0, 0] = 0.5  # lightning_density rises from 0 to 0.5
    text = explain_cell(history, row=0, col=0)
    assert "lightning" in text.lower()


def test_explain_cell_falls_back_when_nothing_rises():
    history = np.full((2, 5, 1, 1), 0.5, dtype=np.float32)  # completely flat
    text = explain_cell(history, row=0, col=0)
    assert "pattern recognition" in text.lower() or "driven" in text.lower()


def test_explain_cell_names_top_two_when_two_rise():
    history = np.zeros((2, 5, 1, 1), dtype=np.float32)
    history[1, 3, 0, 0] = 0.6  # cape
    history[1, 2, 0, 0] = 0.4  # lightning_density
    text = explain_cell(history, row=0, col=0, top_n=2)
    assert "cape" in text.lower()
    assert "lightning" in text.lower()
