# HPC Training Pipeline

Self-contained folder to ship to HPC and train the real nowcast model at a scale
the local 8GB-RAM dev machine can't handle. Everything here is standalone —
no dependency on `backend/`'s FastAPI service, just PyTorch + the data pipeline.

## What's verified vs. what isn't

**Verified against live data on 2026-09-29** (actually run, not guessed):
- `data_prep/download_sevir.py` — SEVIR's real S3 bucket structure, file sizes,
  and the greedy file-selection algorithm (confirmed: one vil/ir069/ir107
  quarterly chunk + 4 matching lght monthly files = 2,157 full-coverage real
  storm events, ~32GB total download).
- `data_prep/preprocess.py` — vil/ir107/lght encoding (byte-scale, Celsius×100,
  point-flash records respectively) and the full extraction pipeline, tested
  end-to-end on 2 real events via remote streaming: correctly produced high
  radar+lightning+cold-cloud-top values for an active storm event and
  near-zero values for a calm event.
- `model/train.py` — smoke-tested locally (synthetic data source, CPU).

**Not verified** (documented honestly, not silently assumed):
- `data_prep/download_era5.py` — API format checked against current docs, but
  needs your own CDS account/token to actually run; never executed end-to-end.
- The cape/shear channels are currently a constant 0.5 placeholder in
  `preprocess.py` — wiring in real ERA5 CAPE/shear per-event is not done (see
  "Next steps" below).

## Folder layout

```
hpc_train/
├── requirements.txt              # torch, h5py, s3fs, cdsapi, etc. — no FastAPI
├── data_prep/
│   ├── download_sevir.py         # verified — downloads a real-data training subset
│   ├── download_era5.py          # unverified — needs your own CDS token
│   └── preprocess.py             # verified — builds training-ready .npz events
├── model/
│   ├── config.py                 # grid/channel contract (mirrors backend/app/core/config.py)
│   ├── synthetic_generator.py    # copy of backend's synthetic storm simulator
│   ├── model.py                  # ConvLSTM (copy of backend/app/ml/model.py)
│   ├── metrics.py                # POD/FAR/CSI/HSS (copy of backend/app/ml/metrics.py)
│   ├── dataset.py                # Synthetic / Real / Mixed torch Datasets
│   └── train.py                  # standalone training entrypoint, file+console logging
├── data/{raw,processed}/         # gitignored — populated by data_prep scripts
├── checkpoints/                  # gitignored — trained model output
├── logs/                         # gitignored — SLURM + train.py logs
├── 01_download_and_preprocess.sbatch   # CPU-only SLURM job
└── 02_train.sbatch                      # GPU SLURM job
```

## Workflow

1. **Edit the two `.sbatch` files**: replace `/home/YOUR_USERNAME/thunderstorm_alert`
   with your actual path, and adjust `--partition`/`--gres`/`--time` for your
   cluster's actual queue names (the values here — `partition=gpu`,
   `gres=shard:20`, `module load cuda-12.8` — match the sample job script
   style provided, but confirm partition/GPU-sharing names are the same for
   this job type on your cluster).

2. **Install dependencies** (once, in your HPC Python env):
   ```bash
   pip install -r requirements.txt
   ```

3. **Download + preprocess real data** (CPU job, no GPU needed):
   ```bash
   sbatch 01_download_and_preprocess.sbatch
   ```
   This downloads ~32GB from SEVIR's public S3 bucket and produces ~2,157
   training-ready `.npz` event files under `data/processed/sevir/`. Increase
   `--min-events` in the sbatch script if you want more (see script comments
   — it'll need to search beyond the single anchor file, which the current
   `select_files()` doesn't do automatically yet, see "Next steps").

4. **Train** (GPU job):
   ```bash
   sbatch 02_train.sbatch
   ```
   Default config: 60 epochs, mixed real+synthetic data, batch size 32,
   hidden channels 32/64 (2x wider than the local dev checkpoint). Adjust
   `--data-source real` to train on real data only once you have enough of it,
   or `--data-source synthetic` to skip the SEVIR download entirely (matches
   what already runs on the dev machine, just faster/bigger).

5. **Retrieve the checkpoint**:
   ```bash
   scp your-hpc:/path/to/hpc_train/checkpoints/convlstm_hpc.pt <local repo>/models/checkpoints/convlstm.pt
   ```
   The main repo's `backend/app/ml/infer.py` reads `hidden_channels` from the
   checkpoint automatically, so a bigger HPC-trained model drops in with zero
   code changes to the backend/frontend.

6. **Re-run evaluation locally**:
   ```bash
   cd ../backend && ../.venv/Scripts/python ../scripts/evaluate.py
   ```
   Check `docs/evaluation_results.md` — the local checkpoint currently loses
   to the optical-flow baseline (see that file + `NEXT_STEPS.md`); the goal
   of this HPC run is to close that gap with real data + more compute.

## Next steps / known gaps

- **ERA5 cape/shear join**: `download_era5.py` downloads the right fields but
  `preprocess.py` doesn't yet look them up per-event (still a 0.5 constant).
  To wire this in: match each event's `time_utc` + lat/lon bbox (from
  CATALOG.csv) to the nearest ERA5 grid cell/timestep in the downloaded
  NetCDF, resize to 64x64 same as the other channels.
- **More than ~2,157 events**: `select_files()` in `download_sevir.py` only
  tries a single anchor file. To scale further, extend it to try multiple
  anchor candidates (the catalog has 12,863 full-coverage events total across
  all years/quarters) and merge their covered-event sets.
- **MOSDAC / real IMD data**: not attempted — MOSDAC requires manual account
  registration + approval (no open API for anonymous bulk access, see
  `docs/research_notes.md` in the main repo). SEVIR is the pragmatic real-data
  stand-in; swapping in real Indian radar/satellite data is future work once
  MOSDAC/IMD data-sharing access is arranged.
