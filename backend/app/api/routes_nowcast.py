from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Query

from app.core.config import CHANNEL_UNITS, CHANNELS, DEFAULT_HORIZON, GRID_BOUNDS, GRID_H, GRID_W, TIME_STEP_MIN, LIGHTNING_OCCURRENCE_THRESHOLD
from app.core.state import state
from app.ml.metrics import scores_over_leadtimes
from app.api.schemas import AlertsResponse, NowcastResponse, ResetRequest, StateResponse

router = APIRouter(prefix="/api", tags=["nowcast"])


@router.get("/state", response_model=StateResponse)
def get_state() -> StateResponse:
    frame = state.latest_frame()
    return StateResponse(
        timestamp=state.current_time.isoformat(),
        bounds=GRID_BOUNDS,
        grid={"h": GRID_H, "w": GRID_W},
        channels=CHANNELS,
        frames={c: frame[c].tolist() for c in CHANNELS},
        units=CHANNEL_UNITS,
    )


@router.get("/nowcast", response_model=NowcastResponse)
def get_nowcast(horizon: int = Query(DEFAULT_HORIZON, ge=1, le=24), model: str = Query("convlstm", pattern="^(convlstm|baseline)$")) -> NowcastResponse:
    history_stack = state.history_stack()
    radar_forecast, lightning_forecast = state.engine.predict(history_stack, model=model)
    radar_forecast, lightning_forecast = radar_forecast[:horizon], lightning_forecast[:horizon]

    radar_truth, lightning_density_truth = state.future_truth(horizon)
    lightning_truth_binary = (lightning_density_truth > LIGHTNING_OCCURRENCE_THRESHOLD).astype(float)

    radar_scores = scores_over_leadtimes(radar_forecast, radar_truth, threshold=0.4)
    lightning_scores = scores_over_leadtimes(lightning_forecast, lightning_truth_binary, threshold=0.5)
    # Headline scores reported to the frontend chart: average of radar-field
    # skill and lightning-occurrence skill per lead time (both matter per the
    # PS's dual "thunderstorm AND lightning" framing, see docs/design.md 2.2).
    combined = {
        k: [round((a + b) / 2, 4) for a, b in zip(radar_scores[k], lightning_scores[k])]
        for k in ("pod", "far", "csi", "hss")
    }

    lead_times = [TIME_STEP_MIN * (i + 1) for i in range(horizon)]

    return NowcastResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        model=model,
        lead_times_min=lead_times,
        radar_forecast=radar_forecast.tolist(),
        lightning_probability=lightning_forecast.tolist(),
        scores=combined,
    )


@router.post("/simulate/step", response_model=StateResponse)
def simulate_step() -> StateResponse:
    state.step()
    return get_state()


@router.post("/simulate/reset", response_model=StateResponse)
def simulate_reset(body: ResetRequest = ResetRequest()) -> StateResponse:
    state.reset(seed=body.seed, num_cells=body.num_cells)
    return get_state()
