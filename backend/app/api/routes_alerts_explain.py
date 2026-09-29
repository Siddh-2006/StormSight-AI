"""Additive alert endpoints: `/api/alerts/explain` and `/api/alerts/cap`.

These sit ALONGSIDE the existing, frozen-contract `/api/alerts`
(docs/technical_flow.md) — they are new URLs, not a change to that
endpoint's response shape, added specifically to close two gaps between the
pitch deck and the working prototype:

1. The "Why this alert?" explanation panel described in the pitch deck
   (Probability & "Why" Panel feature block) — `/explain` adds a `why`
   field per alert, computed by `app.ml.alert_explain.explain_cell`.
2. The "drafted, forecaster-approved, CAP-format alert" output — `/cap`
   wraps the same alerts into a real CAP 1.2 XML document via
   `app.api.cap_format.build_cap_xml`.

Kept in a separate router file (rather than added to routes_alerts.py)
so the original, already-tested alerts route stays exactly as it was —
consistent with how virtual_radar_fill.py, source_dropout.py and
calibration.py were each added as their own isolated module rather than
edited into an existing, working file. See NEXT_STEPS.md for the running,
honest list of what in this repository is demo-grade vs. production-grade;
everything reachable through this router is demo-grade: a real, working,
testable first pass, not a claim that it's ready for actual public alert
dispatch.
"""
from __future__ import annotations

from fastapi import APIRouter, Query, Response

from app.api.cap_format import build_cap_xml
from app.api.schemas import ExplainedAlertEntry, ExplainedAlertsResponse
from app.core.config import TIME_STEP_MIN
from app.core.state import state
from app.data.grid import cell_to_latlon
from app.ml.alert_explain import explain_cell

router = APIRouter(prefix="/api/alerts", tags=["alerts-explain"])

_CONFIDENCE_NOTE = (
    "Raw model probability from the synthetic-data ConvLSTM checkpoint — not yet "
    "conformally calibrated against real observations. See "
    "backend/app/ml/calibration.py (fit_conformal_calibrator) and NEXT_STEPS.md."
)


def _compute_explained_alerts(threshold: float) -> list[ExplainedAlertEntry]:
    history_stack = state.history_stack()
    _, lightning_forecast = state.engine.predict(history_stack, model="convlstm")
    horizon, h, w = lightning_forecast.shape

    alerts: list[ExplainedAlertEntry] = []
    for row in range(h):
        for col in range(w):
            series = lightning_forecast[:, row, col]
            for t in range(horizon):
                if series[t] >= threshold:
                    lat, lon = cell_to_latlon(row, col)
                    alerts.append(
                        ExplainedAlertEntry(
                            cell=(row, col),
                            lat=lat,
                            lon=lon,
                            probability=round(float(series[t]), 4),
                            lead_time_min=TIME_STEP_MIN * (t + 1),
                            why=explain_cell(history_stack, row, col),
                            confidence_note=_CONFIDENCE_NOTE,
                        )
                    )
                    break  # earliest lead time this cell crosses the threshold, same rule as routes_alerts.py
    alerts.sort(key=lambda a: a.probability, reverse=True)
    return alerts


@router.get("/explain", response_model=ExplainedAlertsResponse)
def get_alerts_explained(threshold: float = Query(0.6, ge=0, le=1)) -> ExplainedAlertsResponse:
    return ExplainedAlertsResponse(threshold=threshold, alerts=_compute_explained_alerts(threshold))


@router.get("/cap")
def get_alerts_cap(threshold: float = Query(0.6, ge=0, le=1)) -> Response:
    alerts = _compute_explained_alerts(threshold)
    xml_str = build_cap_xml([a.model_dump() for a in alerts])
    return Response(content=xml_str, media_type="application/xml")
