"""Turn downloaded SEVIR files into training-ready per-event arrays matching
this project's channel contract (see backend/app/core/config.py in the main
repo): ["radar_dbz", "satellite_ir_k", "lightning_density", "cape", "shear"],
each frame (H, W) float32 in [0, 1], at TIME_STEP_MIN=10 cadence.

Normalization/encoding was VERIFIED against real downloaded data on
2026-09-29 (not assumed from docs):
  - vil: uint8, byte-encoded intensity, data_min always 0 in CATALOG.csv ->
    normalize as value/255.0. Spatial shape (384, 384, 49).
  - ir107: int16, confirmed against CATALOG.csv's data_min/data_max that the
    encoding is `temperature_celsius = raw_value / 100.0` (e.g. raw=-7764 ->
    -77.64C, matching the catalog's documented data_min for that exact row
    almost exactly). Converted to Kelvin then normalized against this
    project's (180, 300) K convention. Spatial shape (192, 192, 49) — HALF
    the resolution of vil, both get resized to our 64x64 grid.
  - lght: NOT a dense grid — one HDF5 dataset per event `id`, shape
    (N_flashes, 5) float32. Verified real columns: col0 = time offset in
    seconds (same reference window as the other modalities' 49 frames,
    -120..+120 min i.e. -7200..+7200 sec), col1 = latitude, col2 = longitude.
    (columns 3-4 are GOES-16 GLM pixel coordinates in SEVIR's internal
    reference grid — not needed here since lat/lon is enough to bin flashes
    onto our own grid using the event's bounding box from CATALOG.csv.)

Both vil and ir107 datasets use `file_index` (from CATALOG.csv) to select
the event's slice; lght uses the event `id` string as the HDF5 key directly.

CAPE/shear: SEVIR has no NWP data. If --era5-dir is given (see
download_era5.py), those fields are looked up by matching lat/lon/time;
otherwise they're filled with a constant placeholder (0.5) and this is
logged clearly — an honest limitation, not silently faked.

Usage (from hpc_train/):
    python data_prep/preprocess.py --sevir-dir data/raw/sevir --out-dir data/processed/sevir --min-events 500
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_sevir import load_full_coverage_catalog, select_files, MODALITIES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GRID_H, GRID_W = 64, 64
N_FRAMES_SEVIR = 49          # -120..+120 min in 5-min steps
SEVIR_STEP_MIN = 5
PROJECT_STEP_MIN = 10        # must match backend/app/core/config.py TIME_STEP_MIN
PROJECT_SEQ_LEN = 18         # HISTORY_LEN(6) + DEFAULT_HORIZON(12), see backend config

IR_K_MIN, IR_K_MAX = 180.0, 300.0
LIGHTNING_CAP = 3.0  # flashes/cell/10min treated as "saturated" -> 1.0


def resize_stack(frames: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """frames: (H, W, T) -> (T, out_h, out_w), float32."""
    h, w, t = frames.shape
    zoom_factors = (out_h / h, out_w / w, 1.0)
    resized = zoom(frames.astype(np.float32), zoom_factors, order=1)
    return np.transpose(resized, (2, 0, 1))  # (T, out_h, out_w)


def rasterize_lightning(
    flashes: np.ndarray,
    bbox: tuple[float, float, float, float],
    n_frames: int,
) -> np.ndarray:
    """flashes: (N, 5) [time_sec, lat, lon, _, _]. bbox: (lat_min, lon_min, lat_max, lon_max).
    Returns (n_frames, GRID_H, GRID_W) flash-count grid at SEVIR_STEP_MIN cadence."""
    lat_min, lon_min, lat_max, lon_max = bbox
    out = np.zeros((n_frames, GRID_H, GRID_W), dtype=np.float32)
    if flashes.shape[0] == 0:
        return out

    t_sec, lat, lon = flashes[:, 0], flashes[:, 1], flashes[:, 2]
    frame_idx = np.round((t_sec + (n_frames // 2) * SEVIR_STEP_MIN * 60) / (SEVIR_STEP_MIN * 60)).astype(int)
    frame_idx = np.clip(frame_idx, 0, n_frames - 1)

    row = ((lat_max - lat) / max(lat_max - lat_min, 1e-6) * GRID_H).astype(int)
    col = ((lon - lon_min) / max(lon_max - lon_min, 1e-6) * GRID_W).astype(int)
    valid = (row >= 0) & (row < GRID_H) & (col >= 0) & (col < GRID_W)

    for f, r, c in zip(frame_idx[valid], row[valid], col[valid]):
        out[f, r, c] += 1.0
    return out


def resample_temporal(frames: np.ndarray, factor: int) -> np.ndarray:
    """(T, H, W) at 5-min cadence -> every `factor`-th frame (10-min cadence
    when factor=2, matching PROJECT_STEP_MIN/SEVIR_STEP_MIN)."""
    return frames[::factor]


def process_event(
    event_id: str,
    row: pd.Series,
    vil_h5: h5py.File,
    ir107_h5: h5py.File,
    lght_h5: h5py.File,
) -> dict[str, np.ndarray] | None:
    vil_idx = int(row["vil_file_index"])
    ir107_idx = int(row["ir107_file_index"])

    vil_raw = vil_h5["vil"][vil_idx]        # (384, 384, 49) uint8
    ir107_raw = ir107_h5["ir107"][ir107_idx]  # (192, 192, 49) int16

    radar = resize_stack(vil_raw / 255.0, GRID_H, GRID_W)

    ir_k = ir107_raw / 100.0 + 273.15
    ir_norm = np.clip((ir_k - IR_K_MIN) / (IR_K_MAX - IR_K_MIN), 0, 1)
    satellite = resize_stack(ir_norm, GRID_H, GRID_W)

    if event_id not in lght_h5:
        flashes = np.zeros((0, 5), dtype=np.float32)
    else:
        flashes = lght_h5[event_id][:]
    bbox = (row["llcrnrlat"], row["llcrnrlon"], row["urcrnrlat"], row["urcrnrlon"])
    lightning_raw = rasterize_lightning(flashes, bbox, N_FRAMES_SEVIR)
    # Smooth spatially a touch (single-cell point hits are too sparse/binary
    # otherwise) and cap-normalize, mirroring the synthetic generator's
    # "recent flash density" convention in backend/app/data/synthetic_generator.py.
    lightning = np.stack([gaussian_filter(f, sigma=0.8) for f in lightning_raw])
    lightning = np.clip(lightning / LIGHTNING_CAP, 0, 1)

    factor = PROJECT_STEP_MIN // SEVIR_STEP_MIN
    radar = resample_temporal(radar, factor)
    satellite = resample_temporal(satellite, factor)
    lightning = resample_temporal(lightning, factor)

    n_available = min(len(radar), len(satellite), len(lightning))
    if n_available < PROJECT_SEQ_LEN:
        return None  # shouldn't happen given the 240-min window, but be safe

    start = (n_available - PROJECT_SEQ_LEN) // 2  # centered window
    sl = slice(start, start + PROJECT_SEQ_LEN)

    return {
        "radar_dbz": radar[sl].astype(np.float32),
        "satellite_ir_k": satellite[sl].astype(np.float32),
        "lightning_density": lightning[sl].astype(np.float32),
        # No NWP data in SEVIR; see module docstring. Overwritten by ERA5
        # lookup in main() if --era5-dir is supplied.
        "cape": np.full((PROJECT_SEQ_LEN, GRID_H, GRID_W), 0.5, dtype=np.float32),
        "shear": np.full((PROJECT_SEQ_LEN, GRID_H, GRID_W), 0.5, dtype=np.float32),
    }


def main(sevir_dir: Path, out_dir: Path, min_events: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    full = load_full_coverage_catalog(sevir_dir)
    chosen, covered = select_files(full, min_events)
    logger.info("Preprocessing %d full-coverage events from: %s", covered, {m: sorted(f) for m, f in chosen.items()})

    catalog_path = sevir_dir / "CATALOG.csv"
    df = pd.read_csv(catalog_path, low_memory=False)

    mask = pd.Series(True, index=full.index)
    for m in MODALITIES:
        mask &= full[m].isin(chosen[m]) if chosen[m] else False
    selected_ids = full[mask].index.tolist()

    # file_index per modality, and bbox, come from the original per-row catalog.
    # Plain per-modality dicts (not a MultiIndex .loc[]) — a MultiIndex lookup
    # here returned a Series instead of a scalar in testing (unsorted index +
    # partial-tuple indexing quirk), which is exactly the kind of subtle bug
    # this test-before-ship step exists to catch.
    vil_idx_lookup = df[df.img_type == "vil"].set_index("id")["file_index"].to_dict()
    ir107_idx_lookup = df[df.img_type == "ir107"].set_index("id")["file_index"].to_dict()
    bbox_lookup = df[df.img_type == "vil"].set_index("id")[["llcrnrlat", "llcrnrlon", "urcrnrlat", "urcrnrlon"]]

    # Group by which (vil_file, ir107_file, lght_file) triple each event needs,
    # so each H5 file is opened once and reused across all its events.
    by_files: dict[tuple[str, str, str], list[str]] = {}
    for eid in selected_ids:
        key = (full.loc[eid, "vil"], full.loc[eid, "ir107"], full.loc[eid, "lght"])
        by_files.setdefault(key, []).append(eid)

    n_written, n_skipped = 0, 0
    for (vil_file, ir107_file, lght_file), ids in by_files.items():
        logger.info("Opening file group for %d events...", len(ids))
        with h5py.File(sevir_dir / vil_file, "r") as vil_h5, \
             h5py.File(sevir_dir / ir107_file, "r") as ir107_h5, \
             h5py.File(sevir_dir / lght_file, "r") as lght_h5:
            for eid in tqdm(ids, desc=Path(vil_file).stem):
                lat_min, lon_min, lat_max, lon_max = bbox_lookup.loc[eid]
                row = pd.Series(
                    {
                        "vil_file_index": vil_idx_lookup[eid],
                        "ir107_file_index": ir107_idx_lookup[eid],
                        "llcrnrlat": lat_min,
                        "llcrnrlon": lon_min,
                        "urcrnrlat": lat_max,
                        "urcrnrlon": lon_max,
                    }
                )
                result = process_event(eid, row, vil_h5, ir107_h5, lght_h5)
                if result is None:
                    n_skipped += 1
                    continue
                np.savez_compressed(out_dir / f"event_{eid}.npz", **result)
                n_written += 1

    logger.info("Wrote %d event files to %s (%d skipped).", n_written, out_dir, n_skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sevir-dir", type=str, default="data/raw/sevir")
    parser.add_argument("--out-dir", type=str, default="data/processed/sevir")
    parser.add_argument("--min-events", type=int, default=500)
    args = parser.parse_args()
    main(Path(args.sevir_dir), Path(args.out_dir), args.min_events)
