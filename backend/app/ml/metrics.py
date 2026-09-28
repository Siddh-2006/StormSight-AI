"""Standard nowcasting verification metrics — POD/FAR/CSI/HSS — computed per
lead time at a fixed probability/intensity threshold. This is the same
convention used in the DGMR/NowcastNet papers and Indian nowcasting
verification literature (see docs/research_notes.md section 6)."""
from __future__ import annotations

import numpy as np


def _confusion(pred_binary: np.ndarray, true_binary: np.ndarray) -> tuple[int, int, int, int]:
    hits = int(np.sum((pred_binary == 1) & (true_binary == 1)))
    misses = int(np.sum((pred_binary == 0) & (true_binary == 1)))
    false_alarms = int(np.sum((pred_binary == 1) & (true_binary == 0)))
    correct_negatives = int(np.sum((pred_binary == 0) & (true_binary == 0)))
    return hits, misses, false_alarms, correct_negatives


def pod(hits: int, misses: int) -> float:
    denom = hits + misses
    return hits / denom if denom > 0 else 0.0


def far(hits: int, false_alarms: int) -> float:
    denom = hits + false_alarms
    return false_alarms / denom if denom > 0 else 0.0


def csi(hits: int, misses: int, false_alarms: int) -> float:
    denom = hits + misses + false_alarms
    return hits / denom if denom > 0 else 0.0


def hss(hits: int, misses: int, false_alarms: int, correct_negatives: int) -> float:
    n = hits + misses + false_alarms + correct_negatives
    if n == 0:
        return 0.0
    expected_correct = ((hits + misses) * (hits + false_alarms) + (correct_negatives + misses) * (correct_negatives + false_alarms)) / n
    denom = n - expected_correct
    if denom <= 0:
        return 0.0
    return (hits + correct_negatives - expected_correct) / denom


def scores_for_frame(pred: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, float]:
    pred_b = (pred >= threshold).astype(np.uint8)
    truth_b = (truth >= threshold).astype(np.uint8)
    h, m, fa, cn = _confusion(pred_b, truth_b)
    return {
        "pod": pod(h, m),
        "far": far(h, fa),
        "csi": csi(h, m, fa),
        "hss": hss(h, m, fa, cn),
    }


def scores_over_leadtimes(pred_seq: np.ndarray, truth_seq: np.ndarray, threshold: float) -> dict[str, list[float]]:
    """pred_seq, truth_seq: (T, H, W). Returns lists of scores, one per lead time."""
    out = {"pod": [], "far": [], "csi": [], "hss": []}
    for t in range(pred_seq.shape[0]):
        s = scores_for_frame(pred_seq[t], truth_seq[t], threshold)
        for k in out:
            out[k].append(round(s[k], 4))
    return out
