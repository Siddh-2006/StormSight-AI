"""Shared constants — the single source of truth for the grid/channel contract
documented in docs/technical_flow.md. Frontend and cpp_module must not need to
know these; they only see the API contract."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

GRID_H = 64
GRID_W = 64

GRID_BOUNDS = {
    "lat_min": 19.5,
    "lat_max": 22.5,
    "lon_min": 77.5,
    "lon_max": 80.5,
}

CHANNELS = ["radar_dbz", "satellite_ir_k", "lightning_density", "cape", "shear"]

CHANNEL_UNITS = {
    "radar_dbz": "dBZ (0-70 normalized)",
    "satellite_ir_k": "K (180-300 normalized)",
    "lightning_density": "flashes/cell (0-1 normalized)",
    "cape": "J/kg (0-5000 normalized)",
    "shear": "m/s (0-40 normalized)",
}

TIME_STEP_MIN = 10
HISTORY_LEN = 6          # 1 hour of history fed to the model
DEFAULT_HORIZON = 12     # 2 hours of forecast

# Lightning "occurrence" binary threshold on the normalized lightning_density
# channel, used both for training targets and for alert thresholding.
LIGHTNING_OCCURRENCE_THRESHOLD = 0.15

CHECKPOINT_PATH = str(REPO_ROOT / "models" / "checkpoints" / "convlstm.pt")
