"""Probability calibration — Brier score, reliability-diagram data, and a
simple split-conformal calibration mapping.

Why this file exists (and why it is separate from metrics.py): the pitch deck
and idea write-up for this project describe "calibrated" probabilities
("when we say 70%, it happens 70% of the time"), a Brier score, and
reliability diagrams alongside POD/FAR/CSI/HSS. Until now none of that was
implemented anywhere in this repository — metrics.py only had the four
threshold-based skill scores. This module adds the missing calibration half
honestly and separately, rather than editing metrics.py's already-tested,
already-wired-to-the-frontend public API (see docs/technical_flow.md's
"frozen interface" note). Nothing here is imported by the main FastAPI app
by default — see routes_alerts_explain.py for the one place it IS wired in,
additively, behind a brand-new endpoint that doesn't touch `/api/alerts`.

Terminology, in plain terms:
- Brier score: the mean squared error between a predicted probability and
  the actual 0/1 outcome. Lower is better; 0 is perfect.
- Reliability diagram: bucket predictions by their stated probability (e.g.
  everything predicted "60-70%") and check what fraction of those actually
  occurred. A well-calibrated model's buckets sit close to the diagonal.
- Split-conformal calibration: hold out a calibration set, measure how
  wrong the raw model probabilities are on it, and use that error to widen
  or shift future predictions so the stated probability is trustworthy
  in-distribution. This is a deliberately simple, textbook version (Platt-
  scaling-style monotone correction via isotonic-ish binning), not a
  research-grade conformal predictor — see NEXT_STEPS.md for that honesty
  note carried through consistently with the rest of this project.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def brier_score(pred_prob: np.ndarray, truth_binary: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and 0/1 outcomes.
    Both arrays same shape, values of `pred_prob` in [0, 1]."""
    pred_prob = np.asarray(pred_prob, dtype=np.float64)
    truth_binary = np.asarray(truth_binary, dtype=np.float64)
    return float(np.mean((pred_prob - truth_binary) ** 2))


@dataclass
class ReliabilityBin:
    bin_center: float
    predicted_mean: float
    observed_frequency: float
    count: int


def reliability_diagram(
    pred_prob: np.ndarray, truth_binary: np.ndarray, n_bins: int = 10
) -> list[ReliabilityBin]:
    """Buckets predictions into `n_bins` equal-width probability bins and
    reports, per bin, the mean predicted probability vs. the actual observed
    frequency of the event — the data a reliability diagram plots. Empty
    bins are omitted rather than reported as misleading zeros."""
    pred_prob = np.asarray(pred_prob, dtype=np.float64).ravel()
    truth_binary = np.asarray(truth_binary, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (pred_prob >= lo) & (pred_prob < hi if i < n_bins - 1 else pred_prob <= hi)
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        bins.append(
            ReliabilityBin(
                bin_center=float((lo + hi) / 2),
                predicted_mean=float(np.mean(pred_prob[in_bin])),
                observed_frequency=float(np.mean(truth_binary[in_bin])),
                count=count,
            )
        )
    return bins


@dataclass
class ConformalCalibrator:
    """A fitted, reusable calibration mapping. `apply()` corrects a raw
    model probability using the per-bin observed frequency measured on a
    held-out calibration set — i.e. "when the raw model said ~60%, it was
    actually right 48% of the time, so map future 60%s down toward 48%"."""

    bin_edges: np.ndarray
    bin_correction: np.ndarray  # observed_frequency per bin, same length as bins

    def apply(self, pred_prob: np.ndarray) -> np.ndarray:
        pred_prob = np.asarray(pred_prob, dtype=np.float64)
        bin_idx = np.clip(np.digitize(pred_prob, self.bin_edges[1:-1]), 0, len(self.bin_correction) - 1)
        return self.bin_correction[bin_idx]


def fit_conformal_calibrator(
    cal_pred_prob: np.ndarray, cal_truth_binary: np.ndarray, n_bins: int = 10
) -> ConformalCalibrator:
    """Fits a calibrator on a held-out calibration set (must be disjoint
    from both training and the final evaluation set, or this just measures
    training-set overfitting instead of real calibration error)."""
    bins = reliability_diagram(cal_pred_prob, cal_truth_binary, n_bins=n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Fall back to the identity mapping (mid-bin probability) for any bin
    # that had zero calibration-set examples, so apply() never KeyErrors.
    correction = np.array([(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)])
    for b in bins:
        idx = int(round(b.bin_center * n_bins - 0.5))
        idx = max(0, min(n_bins - 1, idx))
        correction[idx] = b.observed_frequency
    return ConformalCalibrator(bin_edges=edges, bin_correction=correction)
