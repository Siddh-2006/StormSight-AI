# Thunderstorm & Lightning Nowcasting (SIH PS 26072)

AI/ML-based nowcasting of thunderstorms and lightning using (simulated, in this prototype) multi-radar, satellite, lightning-network, and NWP model data — built for **SIH Problem Statement 26072** (Ministry of Earth Sciences / India Meteorological Department, Disaster Management theme).

See `ps.txt` for the original problem statement.

## Docs

- [`docs/design.md`](docs/design.md) — architecture and rationale.
- [`docs/technical_flow.md`](docs/technical_flow.md) — frozen API contract, repo layout, data flow (read this before touching any interface between backend/frontend/cpp_module).
- [`docs/research_notes.md`](docs/research_notes.md) — internet research on SOTA nowcasting models, datasets, and prior art.
- [`NEXT_STEPS.md`](NEXT_STEPS.md) — current status, open questions, what's left.

## Project structure

```
backend/       FastAPI + PyTorch — synthetic data generator, ConvLSTM nowcast model, optical-flow baseline, API
frontend/      React + Vite dashboard (map, timeline, alerts, verification charts)
cpp_module/    pybind11 C++ extension — fast optical-flow baseline (nowcast_cpp)
docs/          design/technical/research docs
data/          raw/processed/synthetic data (gitignored except .gitkeep)
models/        trained model checkpoints (gitignored except .gitkeep)
scripts/       setup scripts
```

## Quick start

No conda — this uses `uv` + `pip` only.

```powershell
# 1. one-shot setup (venv, backend deps, frontend deps)
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
# (or, in Git Bash: bash scripts/setup.sh)

# 2. train a quick synthetic-data model (few minutes on CPU)
cd backend
../.venv/Scripts/python -m app.ml.train --epochs 3

# 3. run the backend API
../.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
# docs at http://localhost:8000/docs

# 4. in another terminal: run the frontend
cd frontend
npm run dev
# open the printed localhost URL
```

The frontend works standalone (mock data) even before the backend is running — see `frontend/NOTES.md`.

### Optional: build the C++ baseline module

```powershell
cd cpp_module
# see cpp_module/README.md for exact verified build commands
```

If `nowcast_cpp` isn't built, the backend automatically falls back to an OpenCV/NumPy implementation of the same optical-flow baseline — nothing breaks either way (see `backend/app/ml/baseline.py`).

## Why synthetic data?

The problem statement did not include a dataset link. The prototype uses a physically-plausible synthetic storm generator (moving/evolving convective cells, CAPE-biased lightning triggering) structured to match real multi-modal feeds (radar + satellite + lightning + NWP), so real data (IMD DWR, MOSDAC INSAT, ERA5 CAPE/shear) can be substituted later behind the same interface without touching the model or frontend. Full rationale in `docs/design.md` §4 and `docs/research_notes.md` §3.
