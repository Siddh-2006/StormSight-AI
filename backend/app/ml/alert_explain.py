"""Simple, honest "why this alert" attribution for one forecast grid cell.

Why this file exists: "Why this alert?" / an explainability panel is one of
the headline UI features in the pitch deck (Probability & "Why" Panel /
"evidence panel shows which signals raised the risk"), and until now the
alert payload (`routes_alerts.py` -> `AlertEntry`) carried nothing but a
probability and a lead time — no attribution at all. `AlertPanel.jsx` has no
explanation UI either. This module supplies the missing logic; wiring it
into a response is done additively in `routes_alerts_explain.py`, leaving
the original `/api/alerts` contract untouched.

The method is deliberately simple and named honestly: it ranks input
channels by how much they rose at that grid cell over the history window
and reports the top contributors in plain language. This is a transparent
heuristic, not a trained saliency/attribution model — the published
research this project cites (Leinonen et al., GRL 2023) uses proper
explainable-AI techniques for this; matching that rigor is future work
(see NEXT_STEPS.md), and this docstring exists so nobody mistakes the
current version for more than it is.
"""
from __future__ import annotations

import numpy as np

from app.core.config import CHANNELS

_PLAIN_LABELS = {
    "radar_dbz": "radar reflectivity",
    "satellite_ir_k": "satellite cloud-top cooling",
    "lightning_density": "lightning strike density",
    "cape": "atmospheric instability (CAPE)",
    "shear": "wind shear",
}


def explain_cell(history_stack: np.ndarray, row: int, col: int, top_n: int = 2) -> str:
    """history_stack: (T, C, H, W) in CHANNELS order (see
    SimulationState.history_stack). Returns one short, plain-language
    sentence naming the input channel(s) that rose the most at (row, col)
    between the first and last frame of the history window."""
    first = history_stack[0, :, row, col]
    last = history_stack[-1, :, row, col]
    delta = last - first  # (C,) — positive means "rising" over the window
    order = np.argsort(delta)[::-1]
    top = [(CHANNELS[i], float(delta[i])) for i in order[:top_n] if delta[i] > 0.01]
    if not top:
        return (
            "Driven by the model's overall spatial pattern recognition rather than "
            "one clearly dominant rising input signal over the last hour."
        )
    parts = [f"rising {_PLAIN_LABELS.get(name, name)} (+{d:.2f} over the window)" for name, d in top]
    return "Driven primarily by " + " and ".join(parts) + "."
