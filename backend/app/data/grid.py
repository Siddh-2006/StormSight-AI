"""Grid <-> lat/lon coordinate helpers for the shared spatial grid."""
from __future__ import annotations

import numpy as np

from app.core.config import GRID_BOUNDS, GRID_H, GRID_W


def cell_to_latlon(row: int, col: int) -> tuple[float, float]:
    """Center lat/lon of grid cell (row, col). Row 0 = north edge (lat_max)."""
    lat = GRID_BOUNDS["lat_max"] - (row + 0.5) * (GRID_BOUNDS["lat_max"] - GRID_BOUNDS["lat_min"]) / GRID_H
    lon = GRID_BOUNDS["lon_min"] + (col + 0.5) * (GRID_BOUNDS["lon_max"] - GRID_BOUNDS["lon_min"]) / GRID_W
    return float(lat), float(lon)


def latlon_grid() -> tuple[np.ndarray, np.ndarray]:
    """Full (H, W) lat and lon meshgrids for the cell centers."""
    rows = np.arange(GRID_H)
    cols = np.arange(GRID_W)
    lat = GRID_BOUNDS["lat_max"] - (rows + 0.5) * (GRID_BOUNDS["lat_max"] - GRID_BOUNDS["lat_min"]) / GRID_H
    lon = GRID_BOUNDS["lon_min"] + (cols + 0.5) * (GRID_BOUNDS["lon_max"] - GRID_BOUNDS["lon_min"]) / GRID_W
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return lat_grid, lon_grid
