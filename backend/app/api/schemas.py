"""Pydantic models mirroring docs/technical_flow.md section 3 exactly. The
frontend is built against those JSON shapes — do not change field names here
without updating that doc and the frontend."""
from __future__ import annotations

from pydantic import BaseModel


class Bounds(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class GridSize(BaseModel):
    h: int
    w: int


class StateResponse(BaseModel):
    timestamp: str
    bounds: Bounds
    grid: GridSize
    channels: list[str]
    frames: dict[str, list[list[float]]]
    units: dict[str, str]


class Scores(BaseModel):
    csi: list[float]
    pod: list[float]
    far: list[float]
    hss: list[float]


class NowcastResponse(BaseModel):
    generated_at: str
    model: str
    lead_times_min: list[int]
    radar_forecast: list[list[list[float]]]
    lightning_probability: list[list[list[float]]]
    scores: Scores | None = None


class AlertEntry(BaseModel):
    cell: tuple[int, int]
    lat: float
    lon: float
    probability: float
    lead_time_min: int


class AlertsResponse(BaseModel):
    threshold: float
    alerts: list[AlertEntry]


class ResetRequest(BaseModel):
    seed: int | None = None
    num_cells: int = 2
