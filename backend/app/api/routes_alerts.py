from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.config import DEFAULT_HORIZON, TIME_STEP_MIN
from app.core.state import state
from app.data.grid import cell_to_latlon
from app.api.schemas import AlertEntry, AlertsResponse

router = APIRouter(prefix="/api", tags=["alerts"])


@router.get("/alerts", response_model=AlertsResponse)
def get_alerts(threshold: float = Query(0.6, ge=0, le=1)) -> AlertsResponse:
    history_stack = state.history_stack()
    _, lightning_forecast = state.engine.predict(history_stack, model="convlstm")
    horizon, h, w = lightning_forecast.shape

    alerts: list[AlertEntry] = []
    for row in range(h):
        for col in range(w):
            series = lightning_forecast[:, row, col]
            for t in range(horizon):
                if series[t] >= threshold:
                    lat, lon = cell_to_latlon(row, col)
                    alerts.append(
                        AlertEntry(
                            cell=(row, col),
                            lat=lat,
                            lon=lon,
                            probability=round(float(series[t]), 4),
                            lead_time_min=TIME_STEP_MIN * (t + 1),
                        )
                    )
                    break  # earliest lead time this cell crosses the threshold

    alerts.sort(key=lambda a: a.probability, reverse=True)
    return AlertsResponse(threshold=threshold, alerts=alerts)
