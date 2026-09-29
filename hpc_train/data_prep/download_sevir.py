"""Download a real-data training subset from SEVIR (MIT Lincoln Laboratory's
Storm EVent ImageRy dataset) for the thunderstorm/lightning nowcasting model.

Why SEVIR: it's the only public dataset that already co-registers radar
(VIL), satellite (IR water-vapor + IR window channels), and lightning (GLM
flash data) for the same storm events — the exact three observation types
named in SIH PS 26072 (plus NWP CAPE/shear, which SEVIR doesn't have — see
download_era5.py for that piece). See docs/research_notes.md in the main
repo for the full literature justification.

Everything below was verified against the LIVE bucket on 2026-09-29, not
assumed from documentation:
  - Bucket: s3://sevir (region us-west-2). Public, anonymous HTTPS GET —
    no AWS account or credentials needed.
  - CATALOG.csv (33MB) lists every event across 5 modalities
    (vis, ir069, ir107, vil, lght) with columns:
    id,file_name,file_index,img_type,time_utc,minute_offsets,episode_id,
    event_id,event_type,llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon,proj,
    size_x,size_y,height_m,width_m,data_min,data_max,pct_missing
  - vil/ir069/ir107 are dense (N_events, 384, 384, 49) uint8/int16 HDF5
    datasets (49 frames = -120..+120 min in 5-min steps); file_index in
    CATALOG.csv is the row along the first axis.
  - lght is structured differently: each H5 file is a flat group of one
    dataset per event `id` (not file_index), shape (N_flashes, 5) float32
    (point/flash records, NOT a dense grid) — see preprocess.py for how
    this gets rasterized onto our project's grid.
  - Per-file sizes are large (a single vil quarterly chunk is ~16GB), so we
    do NOT do random remote access (measured ~56s to stream a single event
    slice over the network — too slow at scale). Instead we compute exactly
    which whole files are needed to cover N full-coverage events, then bulk
    download just those files.
  - Verified concretely: the vil/ir069/ir107 "RANDOMEVENTS_2018_0901_1231"
    quarterly files align perfectly (same 2157 events), while lght uses a
    different (monthly) chunking scheme, so covering that same quarter
    needs 4 separate lght monthly files. The algorithm below discovers this
    generically from the catalog rather than hardcoding it.

Usage (from hpc_train/):
    python data_prep/download_sevir.py --min-events 500 --out-dir data/raw/sevir
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CATALOG_URL = "https://sevir.s3.us-west-2.amazonaws.com/CATALOG.csv"
BUCKET_HTTPS_ROOT = "https://sevir.s3.us-west-2.amazonaws.com"
MODALITIES = ["vil", "ir069", "ir107", "lght"]  # radar-proxy, water vapor, IR window, lightning


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("already have %s, skipping", dest.name)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.rename(dest)


def load_full_coverage_catalog(out_dir: Path) -> pd.DataFrame:
    """Download CATALOG.csv and return the pivot of ids that have ALL of
    MODALITIES present (id -> {modality: file_name})."""
    catalog_path = out_dir / "CATALOG.csv"
    download_file(CATALOG_URL, catalog_path)
    df = pd.read_csv(catalog_path, low_memory=False)
    sub = df[df.img_type.isin(MODALITIES)]
    pivot = sub.pivot_table(index="id", columns="img_type", values="file_name", aggfunc="first")
    full = pivot.dropna(subset=MODALITIES)
    logger.info("Full-coverage candidate events (all of %s present): %d", MODALITIES, len(full))
    return full


def select_files(full: pd.DataFrame, min_events: int) -> tuple[dict[str, set[str]], int]:
    """Greedy selection: anchor on the modality whose single most-common
    file covers the most events, then fill in the other modalities'
    required files (in order of most-events-per-file) until min_events
    is reached or no more coverage is available from this anchor.

    Returns (chosen files per modality, actual events covered)."""
    top_counts = {m: full[m].value_counts() for m in MODALITIES}
    anchor_modality = max(MODALITIES, key=lambda m: top_counts[m].iloc[0])
    anchor_file = top_counts[anchor_modality].index[0]
    logger.info("Anchor modality=%s file=%s (%d events)", anchor_modality, anchor_file, top_counts[anchor_modality].iloc[0])

    chosen: dict[str, set[str]] = {m: set() for m in MODALITIES}
    chosen[anchor_modality] = {anchor_file}
    candidates = full[full[anchor_modality] == anchor_file]

    def coverage_mask() -> pd.Series:
        mask = pd.Series(True, index=full.index)
        for m in MODALITIES:
            mask &= full[m].isin(chosen[m]) if chosen[m] else False
        return mask

    for m in MODALITIES:
        if m == anchor_modality:
            continue
        counts = candidates[m].value_counts()
        for f, _ in counts.items():
            chosen[m].add(f)
            if coverage_mask().sum() >= min_events:
                break

    covered = int(coverage_mask().sum())
    if covered < min_events:
        logger.warning(
            "Could only reach %d full-coverage events from this anchor (requested %d). "
            "This dataset's largest single aligned cluster is capped here — extend "
            "select_files() to try multiple anchors if you need more.",
            covered,
            min_events,
        )
    return chosen, covered


def download_selected(chosen: dict[str, set[str]], out_dir: Path) -> None:
    total_files = sum(len(v) for v in chosen.values())
    logger.info("Downloading %d files...", total_files)
    for modality, files in chosen.items():
        for file_name in files:
            url = f"{BUCKET_HTTPS_ROOT}/data/{file_name}"
            dest = out_dir / file_name
            download_file(url, dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-events", type=int, default=500, help="Minimum full-coverage events to download.")
    parser.add_argument("--out-dir", type=str, default="data/raw/sevir")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = load_full_coverage_catalog(out_dir)
    chosen, covered = select_files(full, args.min_events)
    logger.info("Selected files: %s", {m: sorted(f) for m, f in chosen.items()})
    logger.info("Will cover %d full-coverage events.", covered)
    download_selected(chosen, out_dir)
    logger.info("Done. Run preprocess.py next to build training-ready arrays.")
