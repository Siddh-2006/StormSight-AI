import numpy as np

from app.ml.calibration import brier_score, fit_conformal_calibrator, reliability_diagram


def test_brier_score_perfect():
    pred = np.array([1.0, 0.0, 1.0, 0.0])
    truth = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(pred, truth) == 0.0


def test_brier_score_worst_case():
    pred = np.array([1.0, 0.0])
    truth = np.array([0.0, 1.0])
    assert brier_score(pred, truth) == 1.0


def test_reliability_diagram_well_calibrated():
    rng = np.random.default_rng(0)
    # Predictions that ARE their own true occurrence rate: for each of many
    # samples at probability p, the outcome is 1 with probability p.
    preds = rng.uniform(0, 1, size=5000)
    truth = (rng.uniform(0, 1, size=5000) < preds).astype(float)
    bins = reliability_diagram(preds, truth, n_bins=10)
    assert len(bins) > 0
    for b in bins:
        # Observed frequency should track the predicted mean reasonably
        # closely for a genuinely well-calibrated source (loose tolerance —
        # this is a statistical check on random data, not an exact one).
        assert abs(b.observed_frequency - b.predicted_mean) < 0.15


def test_conformal_calibrator_corrects_overconfidence():
    rng = np.random.default_rng(1)
    # A badly overconfident source: model says 0.9 but is only right half
    # the time.
    n = 2000
    cal_pred = np.full(n, 0.9)
    cal_truth = (rng.uniform(0, 1, size=n) < 0.5).astype(float)

    calibrator = fit_conformal_calibrator(cal_pred, cal_truth, n_bins=10)
    corrected = calibrator.apply(np.array([0.9]))
    assert corrected[0] < 0.7  # pulled well down toward the true ~0.5 rate


def test_conformal_calibrator_handles_unseen_bins():
    # Calibration set never visits low probabilities — apply() must still
    # return something sane (the bin midpoint fallback) rather than error.
    calibrator = fit_conformal_calibrator(np.full(50, 0.9), np.ones(50), n_bins=10)
    corrected = calibrator.apply(np.array([0.05]))
    assert 0.0 <= corrected[0] <= 1.0
