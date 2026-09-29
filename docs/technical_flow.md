# Technical Flow & API Contract

Companion to `docs/design.md`. This is the frozen interface — frontend and the C++ module are built against this contract so backend internals can change without breaking either.

## 1. Repository layout

```
sih_submission2/
├── ps.txt                        # original problem statement
├── docs/
│   ├── design.md
│   ├── technical_flow.md         # this file
│   └── research_notes.md         # internet research on SOTA + datasets
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py               # FastAPI app entrypoint
│   │   ├── core/config.py        # grid size, channel defs, thresholds
│   │   ├── data/
│   │   │   ├── grid.py           # shared grid/coord definitions
│   │   │   └── synthetic_generator.py
│   │   ├── ml/
│   │   │   ├── model.py          # ConvLSTM encoder-decoder
│   │   │   ├── dataset.py
│   │   │   ├── train.py
│   │   │   ├── infer.py
│   │   │   ├── metrics.py        # CSI/POD/FAR/HSS
│   │   │   ├── baseline.py       # optical-flow baseline (calls cpp_module if built, else OpenCV/py fallback)
│   │   │   ├── calibration.py    # additive: Brier/reliability/conformal calibration (see docs/buzzword_coverage.md)
│   │   │   ├── source_dropout.py # additive: availability mask + source-dropout augmentation (opt-in)
│   │   │   ├── virtual_radar_fill.py  # additive: standalone GREMLIN-style U-Net, satellite+lightning -> synthetic radar
│   │   │   └── alert_explain.py  # additive: rule-based "why this alert" attribution
│   │   └── api/
│   │       ├── schemas.py        # pydantic models = the contract below
│   │       ├── routes_nowcast.py
│   │       ├── routes_alerts.py
│   │       ├── routes_alerts_explain.py  # additive: /api/alerts/explain, /api/alerts/cap
│   │       └── cap_format.py     # additive: CAP 1.2 XML builder
│   └── tests/
├── cpp_module/
│   ├── CMakeLists.txt
│   ├── include/advect.hpp
│   └── src/advect.cpp            # pybind11 module `nowcast_cpp`
├── frontend/                     # React + Vite app
├── data/{raw,processed,synthetic}/
├── models/checkpoints/
└── scripts/                      # setup + dataset generation entrypoints
```

## 2. Shared grid convention

- Grid: `H=64, W=64` cells.
- Default demo bounding box: a representative thunderstorm-prone Indian region, e.g. Vidarbha/Nagpur area: `lat ∈ [19.5, 22.5], lon ∈ [77.5, 80.5]` (configurable in `backend/app/core/config.py`, `GRID_BOUNDS`).
- Time step: 10 minutes. History window: 6 steps (1h). Forecast horizon: up to 12 steps (2h).
- Channels, in fixed order (index matters — see `core/config.py:CHANNELS`):
  `["radar_dbz", "satellite_ir_k", "lightning_density", "cape", "shear"]`
- All channel values sent over the API are **normalized floats in [0, 1]** unless stated otherwise; the backend carries the denormalization factors internally so the frontend never needs physical units to render, only for labels/tooltips (which the API also supplies, see below).

## 3. REST API contract

Base URL: `http://localhost:8000/api`

### `GET /api/state`
Current simulated/ingested state (the most recent observed frame stack). Response:

```json
{
  "timestamp": "2026-09-28T10:30:00Z",
  "bounds": { "lat_min": 19.5, "lat_max": 22.5, "lon_min": 77.5, "lon_max": 80.5 },
  "grid": { "h": 64, "w": 64 },
  "channels": ["radar_dbz", "satellite_ir_k", "lightning_density", "cape", "shear"],
  "frames": {
    "radar_dbz": [[0.1, 0.2, ...], ...],
    "satellite_ir_k": [[...]],
    "lightning_density": [[...]],
    "cape": [[...]],
    "shear": [[...]]
  },
  "units": { "radar_dbz": "dBZ (0-70 normalized)", "satellite_ir_k": "K (180-300 normalized)", "cape": "J/kg (0-5000 normalized)", "shear": "m/s (0-40 normalized)" }
}
```

### `GET /api/nowcast?horizon=12&model=convlstm|baseline`
Runs inference from the current state. Response:

```json
{
  "generated_at": "2026-09-28T10:30:00Z",
  "model": "convlstm",
  "lead_times_min": [10, 20, 30, ..., 120],
  "radar_forecast": [ [[...64x64...]], ... ],
  "lightning_probability": [ [[...64x64...]], ... ],
  "scores": {
    "csi": [0.61, 0.55, 0.48, ...],
    "pod": [0.7, 0.65, ...],
    "far": [0.2, 0.25, ...],
    "hss": [0.5, 0.44, ...]
  }
}
```
`radar_forecast` / `lightning_probability` are arrays indexed `[lead_time_index][row][col]`. `scores` are only populated when ground truth is available (synthetic replay mode, where future truth exists) — real-time deployment would omit or lag them.

### `GET /api/alerts?threshold=0.6`
Thresholded lightning risk, aggregated to a coarse cell/region list for a simple "watch list" UI panel:

```json
{
  "threshold": 0.6,
  "alerts": [
    { "cell": [12, 34], "lat": 20.85, "lon": 78.9, "probability": 0.82, "lead_time_min": 30 }
  ]
}
```

### Additive endpoints (not part of the frozen contract above)

Everything below was added in a later, isolated pass to close specific gaps
between the pitch deck and this prototype (see `docs/buzzword_coverage.md`
for the full traceability table). They are new URLs under `/api/alerts/`,
not changes to `GET /api/alerts` above — its response shape is unchanged
and still exactly as documented. Frontend and `cpp_module` were never built
against these, so they are free to change without the "frozen interface"
concerns that apply to section 3 proper.

#### `GET /api/alerts/explain?threshold=0.6`
Same alerts as `GET /api/alerts`, plus a `why` field (a short, honestly-labelled,
rule-based explanation — see `backend/app/ml/alert_explain.py`) and a
`confidence_note` field stating plainly that the probability is a raw model
output, not yet conformally calibrated:

```json
{
  "threshold": 0.6,
  "alerts": [
    {
      "cell": [12, 34], "lat": 20.85, "lon": 78.9,
      "probability": 0.82, "lead_time_min": 30,
      "why": "Driven primarily by rising lightning strike density (+0.40 over the window).",
      "confidence_note": "Raw model probability from the synthetic-data ConvLSTM checkpoint — not yet conformally calibrated against real observations. See backend/app/ml/calibration.py (fit_conformal_calibrator) and NEXT_STEPS.md."
    }
  ]
}
```

#### `GET /api/alerts/cap?threshold=0.6`
The same alerts, wrapped as a valid CAP 1.2 XML document (`application/xml`),
built by `backend/app/api/cap_format.py`. `status` is always `Draft` — this
prototype never claims to dispatch a real public alert.

### `POST /api/simulate/step`
Advances the synthetic generator by one time step (demo control — lets the frontend "play" a live feed without needing a real ingestion pipeline). No body required. Returns the same shape as `GET /api/state`.

### `POST /api/simulate/reset`
Resets the synthetic generator to a fresh random storm scenario. Optional body: `{ "seed": 42, "num_cells": 3 }`.

### `WS /ws/stream`
Optional (stretch goal): pushes a `GET /api/state`-shaped message every N seconds during a "live demo" mode, so the frontend can auto-animate without polling.

## 4. C++ module contract (`cpp_module`)

Exposed to Python as module `nowcast_cpp` (built via pybind11 + CMake, importable from the venv). Backend's `backend/app/ml/baseline.py` imports it; if the compiled extension isn't available (e.g. not built yet), it falls back to a pure-Python/OpenCV implementation of the same function signature so the rest of the system still runs.

```python
def advect_frames(frames: np.ndarray, n_steps: int) -> np.ndarray:
    """
    frames: float32 array, shape (T, H, W), most recent frame last.
    n_steps: number of future steps to extrapolate.
    returns: float32 array, shape (n_steps, H, W) — extrapolated via dense
             optical-flow motion vectors estimated from the last two input
             frames, advected forward (semi-Lagrangian).
    """
```

Also expose `estimate_flow(frame_a, frame_b) -> (flow_u, flow_v)` (each `(H, W)` float32) for reuse/testing.

## 5. Frontend contract (what it must render)

Given only the REST contract above, the frontend (`frontend/`) must provide:

1. **Map view** of the bounding box with toggleable overlay layers: radar, satellite IR, observed lightning density, predicted lightning probability — each rendered as a semi-transparent heatmap image over the map (grid → canvas → image overlay, e.g. via `react-leaflet`'s `ImageOverlay` or a canvas layer).
2. **Timeline slider** over `lead_times_min` from `/api/nowcast`, scrubbing which forecast frame is displayed; play/pause auto-advance.
3. **Alert panel** listing `/api/alerts` entries sorted by probability, with a simple severity color scale.
4. **Verification chart** (Recharts line chart) plotting `scores.csi/pod/far/hss` vs. `lead_times_min` — this is the "does the ML model actually beat the baseline" evidence for judges. Should be able to overlay both `model=convlstm` and `model=baseline` results.
5. A "Play live demo" control that calls `POST /api/simulate/step` on an interval and re-fetches `/api/state` + `/api/nowcast`, to simulate a live operational feed for the demo without needing WebSockets.

Styling/library conventions: React + Vite, Tailwind for layout, Recharts for charts (per the dataviz skill's guidance — consistent categorical/sequential palettes, accessible in light/dark).

## 6. Data/control flow (sequence)

1. Backend startup: synthetic generator initializes a random scenario (a few storm cells) → in-memory state buffer holds the last `history_len` frames per channel.
2. Frontend loads → `GET /api/state` to draw current layers → `GET /api/nowcast` to draw forecast + scores.
3. User scrubs timeline → frontend re-renders from already-fetched `radar_forecast`/`lightning_probability` arrays (no refetch needed per scrub).
4. User clicks "Play live demo" → frontend calls `POST /api/simulate/step` every ~2s (demo speed) → re-fetches `/api/state` + `/api/nowcast` → map animates as if live.
5. Alerts panel refreshes alongside `/api/nowcast` calls.

## 7. Environment / tooling

- Python: `uv venv .venv` (done), dependencies via `uv pip install -r backend/requirements.txt` (or `pyproject.toml`).
- No conda — uv/pip only.
- C++: CMake + a C++17 compiler (MSYS2 g++ available), pybind11 fetched via pip (`pybind11[global]`) or vendored.
- Frontend: Node.js (v22 available), `npm create vite@latest`.
- Everything runs locally for the demo; no cloud dependency required.
