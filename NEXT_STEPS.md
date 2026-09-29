# Status & Next Steps

_Last updated: 2026-09-29_

## What's working right now

- **Backend** (`backend/`): FastAPI + PyTorch. Synthetic multi-channel storm generator, ConvLSTM nowcast model (dual-head: radar + lightning probability), optical-flow baseline. 11/11 tests passing. Trained a smoke-test checkpoint (3 epochs, loss decreasing: radar_mse 0.17→0.08).
- **Frontend** (`frontend/`): React + Vite dashboard — map with radar/satellite/lightning layers, timeline scrubber, alert panel, CSI/POD/FAR/HSS verification chart, live-demo mode. Builds clean, runs standalone with mock data or against the real backend.
- **C++ module** (`cpp_module/`): `nowcast_cpp` pybind11 extension for the optical-flow baseline — built, tested, confirmed in use by the backend (not just the OpenCV fallback).
- **Docs**: `docs/design.md` (architecture), `docs/technical_flow.md` (frozen API contract), `docs/research_notes.md` (SOTA models/datasets research).
- Repo pushed to `https://github.com/bhscodz/thunderstorm_alert`.

## Evaluation results (honest finding — see `docs/evaluation_results.md`)

Ran a batch evaluation (`scripts/evaluate.py`, 30 held-out synthetic episodes) comparing the trained ConvLSTM against the optical-flow baseline. **The current checkpoint does not beat the baseline** — the optical-flow baseline wins on radar CSI/HSS at every lead time, and the ConvLSTM's skill collapses to ~0 beyond 60-70 min lead time (likely a decoder training issue — later autoregressive steps get weaker gradient signal with only 8 epochs of training). On the way there, we also caught and fixed a real bug: plain MSE loss let the model collapse to predicting near-zero everywhere (radar reflectivity is heavily skewed toward background pixels); switched to a TrajGRU-style intensity-weighted MSE (`weighted_mse_loss` in `backend/app/ml/model.py`), which fixed the "predicts almost nothing" collapse but didn't yet close the gap with the baseline.

This is a legitimate, undersold-rather-than-oversold result for a proof-of-concept trained for ~3 minutes of wall-clock compute on ~100 synthetic episodes total. The literature (see `docs/research_notes.md`) confirms optical-flow/pySTEPS-style baselines are strong at short lead times and hard to beat without substantial training data — DGMR needed a large real radar archive to reliably beat it. More epochs/episodes and/or real SEVIR-style data would be the expected path to close the gap.

**Codex was tried for this evaluation task first but got stuck in a model-backend reconnection loop for ~50 minutes with zero progress; killed and done directly instead.**

## Known constraints worth knowing about

- **This machine has only 8GB RAM**, shared with multiple VSCode windows, Chrome tabs, and (during development) `cpptools` indexing the new C++ code. This caused real, repeated slowdowns (training that should take ~3 min sometimes took 10x longer under contention) — not a code bug, verified by watching CPU-time-still-climbing vs. truly-stuck. `torch.set_num_threads(4)` in `backend/app/ml/train.py` and `scripts/evaluate.py` avoids a separate thread-oversubscription issue. Closing unused Chrome tabs/VSCode windows measurably speeds up iteration here.
- The model checkpoint (`models/checkpoints/convlstm.pt`) is trained (8 epochs, ~96 synthetic episodes/epoch, weighted-MSE + focal loss) but **does not yet beat the optical-flow baseline** — see the evaluation results above. This is the top technical priority if there's time before judging.

## Suggested next steps (in priority order for a hackathon demo)

1. **Decide whether to close the ConvLSTM-vs-baseline gap before the demo** (see Open Questions) — the honest current state is a real POC weakness, not a blocker, but it affects the pitch.
2. **Run the full stack together** (`uvicorn` + `npm run dev`) and click through the actual UI in a browser — this hasn't been visually verified yet, only build/API-level checks (no browser tool was available in this session).
3. **Prepare the pitch narrative**: lean on `docs/research_notes.md` for credibility (cite SEVIR, MeteoSwiss fusion paper, DGMR/NowcastNet as SOTA context), be upfront that synthetic data was a deliberate, documented choice given no dataset was issued with the PS, and frame the evaluation honestly — the architecture, verification methodology (POD/FAR/CSI/HSS), and pipeline are the deliverable; the specific checkpoint is a proof-of-concept.
4. **Optional polish** (only if time remains): more training epochs/episodes to try to close the baseline gap; WebSocket live streaming instead of polling; district-level polygon alert aggregation instead of per-cell; real SEVIR data loader as a drop-in replacement for the synthetic generator (see `docs/research_notes.md` §3).

## Decisions made (2026-09-29)

- **Real training moves to HPC.** User has a powerful HPC available — see `scripts/HPC_TRAINING.md` (workflow) and `scripts/train_hpc.sbatch` (SLURM template, needs cluster-specific `#SBATCH` edits). The local checkpoint stays a proof-of-concept until an HPC-trained one is retrieved. `backend/app/ml/infer.py` reads `hidden_channels` from the checkpoint automatically, so a bigger HPC-trained model drops in with zero code changes.
- **Fixed a real path bug found while wiring this up**: `CHECKPOINT_PATH` was a relative path that silently resolved under whatever directory a script was run from (ended up at `backend/models/checkpoints/` instead of the documented repo-root `models/checkpoints/`). Now computed from `REPO_ROOT` in `backend/app/core/config.py` — works regardless of cwd. (It was gitignored either way via the generic `*.pt` rule, so nothing leaked.)
- **Codex dropped** for now after its reconnection-loop failure — doing remaining work directly.
- **One-command dev launcher added**: `scripts/run_dev.ps1` (Windows) / `scripts/run_dev.sh` (bash) start backend + frontend together, Ctrl+C stops both. Smoke-tested working.

## Open questions for you

- Any preference on how deep to go on the pitch deck / write-up — want me to draft one?
- When you have HPC access ready, let me know and I'll help fill in `scripts/train_hpc.sbatch`'s cluster-specific placeholders (partition/account/GPU type/walltime) if you share those details.
