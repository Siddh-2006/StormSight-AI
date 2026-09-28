# Design Document — AI/ML Nowcasting of Thunderstorms & Lightning

**SIH Problem Statement 26072** | Ministry of Earth Sciences (MoES) / IMD | Theme: Disaster Management

> Status: living document. Research citations will be filled in as they're gathered (see `docs/research_notes.md` once available). This version captures the architecture we're building against.

## 1. Problem Framing

Predict thunderstorm and lightning occurrence 0–2 hours ahead ("nowcasting", as opposed to numerical weather prediction which is skillful mainly beyond ~6h) by fusing:

- **Radar** — Doppler Weather Radar (DWR) reflectivity (dBZ), IMD operates ~39 DWRs across India.
- **Satellite** — INSAT-3D/3DR IR brightness temperature (cold cloud tops = deep convection proxy).
- **Lightning** — point strike data (lat/lon/time/polarity) from ground-based lightning location networks (India: IITM Lightning Location Network) — sparse, high-value signal.
- **Model (NWP) data** — slow-varying environmental instability fields: CAPE, K-index, wind shear, from GFS/ERA5/NCMRWF regional runs — these say *whether* the atmosphere can support convection, radar/satellite say *where it's happening now*.

The problem statement provides no dataset link, so the prototype must (a) work end-to-end on a **physically-plausible synthetic data generator** we control, and (b) be architected so real feeds (MOSDAC, IMD DWR, ERA5/CDS) can be substituted behind the same interface without touching the model or frontend.

## 2. Core Approach

### 2.1 Common spatiotemporal grid

All four modalities are resampled onto one shared lat/lon grid (default: 64×64 cells at ~0.05–0.1° resolution, i.e. a few-hundred-km regional window) and a common time step (10 min). Point data (lightning strikes) is rasterized into per-cell flash-count/density; NWP fields (which update every few hours) are held constant/interpolated across the fast time steps.

This gives a tensor `X[t, C, H, W]` with channels:

| # | Channel | Source | Update cadence | Role |
|---|---|---|---|---|
| 0 | Radar reflectivity (dBZ, normalized) | DWR / synthetic | 10 min | "where storms are now" |
| 1 | Satellite IR brightness temp (K, normalized) | INSAT-3D / synthetic | 10–15 min | cloud-top / convection depth proxy |
| 2 | Lightning flash density | Ground network / synthetic | 10 min | ground truth for the thing we're predicting |
| 3 | CAPE (J/kg, normalized) | NWP / ERA5 / synthetic | 3–6 h (held constant) | instability context |
| 4 | Wind shear (m/s, normalized) | NWP / ERA5 / synthetic | 3–6 h (held constant) | storm organization context |

### 2.2 Model

**Primary model**: Encoder–decoder ConvLSTM (a compact SmaAt-UNet-style variant: depthwise-separable convs + ConvLSTM bottleneck) — a well-established, hackathon-feasible architecture for radar/precipitation nowcasting (successor family to the original ConvLSTM precipitation nowcasting line of work, and the same family DGMR/NowcastNet build on, but small enough to train on CPU/single GPU in a hackathon timeframe).

- **Input**: last 6 frames (1 hour of history) × 5 channels.
- **Output** (multi-task, two heads sharing the encoder):
  - Future radar reflectivity, next 6–12 frames (1–2h) — regression, MSE/SSIM loss.
  - Future lightning probability, next 6–12 frames — binary/probabilistic, **focal loss** (lightning is a rare-event class; plain BCE under-predicts it).
- Environmental channels (CAPE, shear) are concatenated at the bottleneck as slow-varying context rather than run through the temporal encoder, since they don't evolve frame-to-frame.

**Baseline model** (for comparison, and as a fallback when the ML model isn't confident): dense **optical-flow extrapolation** (Lucas-Kanade/Farneback-style advection) of the last radar frame — classic operational nowcasting technique (the same principle pySTEPS uses operationally). Implemented in C++ (`cpp_module/`) via pybind11, both to demonstrate a non-ML interpretable fallback and to keep the numeric core fast.

### 2.3 Evaluation

Standard nowcasting verification, at multiple reflectivity/probability thresholds and multiple lead times:

- **CSI** (Critical Success Index)
- **POD** (Probability of Detection)
- **FAR** (False Alarm Ratio)
- **HSS** (Heidke Skill Score)

Reported as skill-decay curves (score vs. lead time), and ML vs. optical-flow baseline is the headline comparison for the demo.

## 3. System Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│  Data Sources        │     │  Backend (FastAPI)    │     │  Frontend (React)   │
│  ────────────────    │     │  ──────────────────   │     │  ─────────────────  │
│  Synthetic generator │────▶│  Ingest/grid adapter   │────▶│  Map view (layers:  │
│  (swap-in: DWR/      │     │  ML inference (ConvLSTM│     │  radar/sat/lightning│
│   MOSDAC/ERA5 later) │     │   + C++ baseline)      │◀────┤  /nowcast)          │
│                       │     │  Alerting (threshold   │     │  Timeline slider    │
│                       │     │   per grid cell)       │     │  Alert panel        │
│                       │     │  REST + WebSocket API  │     │  Verification chart │
└─────────────────────┘     └──────────────────────┘     └────────────────────┘
```

- **Backend**: Python 3.13, FastAPI, PyTorch, NumPy/xarray, uvicorn. Owns the synthetic generator, model training/inference, and the API contract in `docs/technical_flow.md`.
- **C++ module**: pybind11 extension exposing the optical-flow baseline advection routine to Python, callable from the backend inference path.
- **Frontend**: React + Vite, react-leaflet for the map, Recharts for verification score charts (per house dataviz conventions), Tailwind for styling. Consumes only the documented REST/WebSocket contract — no backend internals.

## 4. Why synthetic data, and how it stays credible

No real dataset was linked in the PS. The generator (`backend/app/data/synthetic_generator.py`) simulates convective cells as moving/evolving 2D Gaussian "blobs" with:
- random-walk motion + growth/decay lifecycle (storm initiation → mature → dissipation),
- reflectivity core intensity that stochastically triggers lightning flashes once above a threshold (correlated with a synthetic CAPE field, mimicking the real physical link between instability and lightning),
- satellite IR channel derived as inversely correlated with reflectivity core height (colder cloud top over stronger cells).

This is close in spirit to how published nowcasting work validates architectures before swapping to operational feeds, and mirrors the structure of **SEVIR** (MIT Lincoln Lab's Storm EVent ImagRy dataset) — one of the few real, public datasets that already co-locates radar (NEXRAD VIL), satellite (GOES-16 IR/visible) and lightning (GOES-16 GLM) imagery, and is the recommended real-data drop-in to validate against if time permits (see `docs/research_notes.md`).

## 5. Next steps for scope expansion (post-hackathon-prototype)

- Replace synthetic feeds with real MOSDAC (INSAT-3D/3DR) and IMD DWR feeds behind the same grid-adapter interface.
- Add ERA5/NCMRWF CAPE & shear ingestion via Copernicus CDS API.
- District-level polygon aggregation of lightning probability for actionable IMD-style watches/warnings.
- Model upgrade path: attention-based multi-modal fusion, or a generative model (DGMR-style) for ensemble/uncertainty-aware nowcasts.
