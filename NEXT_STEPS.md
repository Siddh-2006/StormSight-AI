# Status & Next Steps

_Last updated: 2026-09-29_

## What's working right now

- **Backend** (`backend/`): FastAPI + PyTorch. Synthetic multi-channel storm generator, ConvLSTM nowcast model (dual-head: radar + lightning probability), optical-flow baseline. 11/11 tests passing. Trained a smoke-test checkpoint (3 epochs, loss decreasing: radar_mse 0.17→0.08).
- **Frontend** (`frontend/`): React + Vite dashboard — map with radar/satellite/lightning layers, timeline scrubber, alert panel, CSI/POD/FAR/HSS verification chart, live-demo mode. Builds clean, runs standalone with mock data or against the real backend.
- **C++ module** (`cpp_module/`): `nowcast_cpp` pybind11 extension for the optical-flow baseline — built, tested, confirmed in use by the backend (not just the OpenCV fallback).
- **Docs**: `docs/design.md` (architecture), `docs/technical_flow.md` (frozen API contract), `docs/research_notes.md` (SOTA models/datasets research).
- Repo pushed to `https://github.com/bhscodz/thunderstorm_alert`.

## In progress

- Codex is running a batch evaluation (`scripts/evaluate.py`) comparing the ConvLSTM model against the optical-flow baseline across ~30 synthetic episodes, producing `docs/evaluation_results.md` + a CSI chart — this is the "does ML actually beat the classical baseline" evidence for judges.

## Known constraints worth knowing about

- **This machine has only 8GB RAM.** Running the training script, a C++ build, and a Codex task concurrently caused severe slowdown (thread oversubscription + memory thrashing) earlier — fixed for training via `torch.set_num_threads(4)`, but keep this in mind if running multiple heavy things at once.
- The model checkpoint (`models/checkpoints/convlstm.pt`) is only lightly trained (3 epochs on ~64 synthetic episodes) — enough to prove the pipeline learns something real, not enough to be a strong model. More epochs/episodes (`python -m app.ml.train --epochs 10 --episodes-per-epoch 200`) would meaningfully improve it if there's time.

## Suggested next steps (in priority order for a hackathon demo)

1. **Review the evaluation results** once Codex finishes — decide if the model needs more training before the demo.
2. **Run the full stack together** (`uvicorn` + `npm run dev`) and click through the actual UI in a browser — this hasn't been visually verified yet, only build/API-level checks (no browser tool was available in this session).
3. **Train longer** if time and machine memory allow — current checkpoint is a proof-of-concept, not a polished model.
4. **Prepare the pitch narrative**: lean on `docs/research_notes.md` for credibility (cite SEVIR, MeteoSwiss fusion paper, DGMR/NowcastNet as SOTA context) and be upfront that synthetic data was a deliberate, documented choice given no dataset was issued with the PS.
5. **Optional polish** (only if time remains): WebSocket live streaming instead of polling; district-level polygon alert aggregation instead of per-cell; longer training run; real SEVIR data loader as a drop-in replacement for the synthetic generator (see `docs/research_notes.md` §3 for why SEVIR is the natural real-data upgrade path).

## Open questions for you

- Do you want more training time invested before the demo, or is the current lightly-trained checkpoint fine given the time budget?
- Should I set up a one-command "run everything" script (backend + frontend together) for the actual demo/judging moment?
- Any preference on how deep to go on the pitch deck / write-up — want me to draft one?
