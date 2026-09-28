import numpy as np

from app.ml.metrics import scores_for_frame, scores_over_leadtimes


def test_perfect_prediction():
    truth = np.array([[0.0, 1.0], [1.0, 0.0]])
    scores = scores_for_frame(truth.copy(), truth, threshold=0.5)
    assert scores["pod"] == 1.0
    assert scores["far"] == 0.0
    assert scores["csi"] == 1.0


def test_all_miss():
    pred = np.zeros((2, 2))
    truth = np.ones((2, 2))
    scores = scores_for_frame(pred, truth, threshold=0.5)
    assert scores["pod"] == 0.0
    assert scores["csi"] == 0.0


def test_scores_over_leadtimes_shape():
    pred = np.random.rand(5, 8, 8)
    truth = np.random.rand(5, 8, 8)
    out = scores_over_leadtimes(pred, truth, threshold=0.5)
    assert set(out.keys()) == {"pod", "far", "csi", "hss"}
    for v in out.values():
        assert len(v) == 5
