"""Standalone copy of backend/app/core/config.py's grid/channel contract, so
hpc_train/ doesn't depend on the backend/ package (no FastAPI needed here).
Keep in sync with backend/app/core/config.py if the contract changes."""
from __future__ import annotations

from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
HPC_TRAIN_ROOT = THIS_DIR.parent

GRID_H = 64
GRID_W = 64

GRID_BOUNDS = {
    "lat_min": 19.5,
    "lat_max": 22.5,
    "lon_min": 77.5,
    "lon_max": 80.5,
}

CHANNELS = ["radar_dbz", "satellite_ir_k", "lightning_density", "cape", "shear"]

TIME_STEP_MIN = 10
HISTORY_LEN = 6
DEFAULT_HORIZON = 12

LIGHTNING_OCCURRENCE_THRESHOLD = 0.15

CHECKPOINT_PATH = str(HPC_TRAIN_ROOT / "checkpoints" / "convlstm_hpc.pt")
PROCESSED_SEVIR_DIR = str(HPC_TRAIN_ROOT / "data" / "processed" / "sevir")
