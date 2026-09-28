"""Physically-plausible synthetic multi-channel storm data generator.

No dataset was provided with the problem statement (see ps.txt / docs/design.md
section 4), so this generator stands in for real radar/satellite/lightning/NWP
feeds. It is deliberately structured to match the channel layout a real feed
(e.g. SEVIR-style radar+satellite+GLM lightning, see docs/research_notes.md)
would provide, so a real data loader can be swapped in later behind the same
interface (`StormGenerator.step()` -> dict[str, np.ndarray]) without touching
the model or API.

Physical story simulated:
- A slow-varying CAPE/shear "environment" field biases where storms form.
- Storm cells are moving, growing-then-decaying 2D Gaussian blobs of radar
  reflectivity.
- Satellite IR brightness temperature is colder (lower normalized value) over
  and around reflectivity cores (deep convection -> cold cloud tops), with a
  wider footprint than the radar core (anvil/cloud shield extends further).
- Lightning flashes are stochastically triggered near a cell's core once its
  intensity crosses a threshold that itself depends on local CAPE (more
  unstable environment -> lower intensity needed to spark lightning), and the
  displayed lightning_density channel is a decaying recent-flash density
  (not a lifetime accumulation), so it look like "current lightning activity".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from app.core.config import CHANNELS, GRID_H, GRID_W


def _gaussian_blob(h: int, w: int, row: float, col: float, radius: float, amplitude: float) -> np.ndarray:
    rr, cc = np.mgrid[0:h, 0:w]
    dist_sq = (rr - row) ** 2 + (cc - col) ** 2
    return amplitude * np.exp(-dist_sq / (2.0 * radius**2))


@dataclass
class StormCell:
    row: float
    col: float
    drow: float
    dcol: float
    radius: float
    peak_intensity: float
    age: int = 0
    lifespan: int = 18  # ~3 hours at 10-min steps

    def lifecycle_factor(self) -> float:
        """Triangular growth->mature->decay profile in [0, 1]."""
        t = self.age / max(self.lifespan, 1)
        if t <= 0.4:
            return t / 0.4
        if t <= 0.6:
            return 1.0
        return max(0.0, 1.0 - (t - 0.6) / 0.4)

    def current_intensity(self) -> float:
        return self.peak_intensity * self.lifecycle_factor()

    def is_alive(self) -> bool:
        return self.age < self.lifespan

    def advance(self) -> None:
        self.row += self.drow
        self.col += self.dcol
        self.drow += np.random.normal(0, 0.08)
        self.dcol += np.random.normal(0, 0.08)
        self.age += 1


@dataclass
class StormGenerator:
    h: int = GRID_H
    w: int = GRID_W
    rng_seed: int | None = None
    cells: list[StormCell] = field(default_factory=list)
    cape_field: np.ndarray = field(default=None)  # type: ignore[assignment]
    shear_field: np.ndarray = field(default=None)  # type: ignore[assignment]
    lightning_density: np.ndarray = field(default=None)  # type: ignore[assignment]
    steps_since_env_update: int = 0
    env_update_interval: int = 18  # regenerate CAPE/shear every ~3h
    max_cells: int = 4
    step_count: int = 0

    def __post_init__(self) -> None:
        self.reset(seed=self.rng_seed, num_cells=2)

    def reset(self, seed: int | None = None, num_cells: int = 2) -> None:
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed if seed is not None else np.random.randint(0, 2**31 - 1))
        self.cells = []
        self.lightning_density = np.zeros((self.h, self.w), dtype=np.float32)
        self.step_count = 0
        self._regen_environment()
        for _ in range(num_cells):
            self._spawn_cell()

    def _regen_environment(self) -> None:
        cape_noise = self.rng.normal(0, 1, size=(self.h, self.w))
        self.cape_field = np.clip(gaussian_filter(cape_noise, sigma=8) * 0.3 + 0.5, 0, 1).astype(np.float32)
        shear_noise = self.rng.normal(0, 1, size=(self.h, self.w))
        self.shear_field = np.clip(gaussian_filter(shear_noise, sigma=10) * 0.25 + 0.4, 0, 1).astype(np.float32)
        self.steps_since_env_update = 0

    def _spawn_cell(self) -> None:
        # Bias spawn location toward high-CAPE regions: sample a handful of
        # candidate points, weight by CAPE, pick the best.
        candidates = self.rng.integers(0, [self.h, self.w], size=(8, 2))
        weights = np.array([self.cape_field[r, c] for r, c in candidates])
        weights = weights / weights.sum() if weights.sum() > 0 else None
        idx = self.rng.choice(len(candidates), p=weights) if weights is not None else 0
        row, col = candidates[idx]
        self.cells.append(
            StormCell(
                row=float(row),
                col=float(col),
                drow=float(self.rng.normal(0, 0.5)),
                dcol=float(self.rng.normal(0.6, 0.5)),  # slight eastward drift bias
                radius=float(self.rng.uniform(3.0, 6.0)),
                peak_intensity=float(self.rng.uniform(0.55, 1.0)),
                lifespan=int(self.rng.integers(10, 24)),
            )
        )

    def step(self) -> dict[str, np.ndarray]:
        """Advance the simulation by one TIME_STEP_MIN and return the current
        multi-channel frame stack as float32 arrays in [0, 1], keyed by
        app.core.config.CHANNELS."""
        self.step_count += 1
        self.steps_since_env_update += 1
        if self.steps_since_env_update >= self.env_update_interval:
            self._regen_environment()

        for cell in self.cells:
            cell.advance()
        self.cells = [c for c in self.cells if c.is_alive() and 0 <= c.row < self.h and 0 <= c.col < self.w]

        spawn_prob = 0.15 * float(np.mean(self.cape_field))
        if len(self.cells) < self.max_cells and self.rng.random() < spawn_prob:
            self._spawn_cell()

        radar = np.zeros((self.h, self.w), dtype=np.float32)
        new_flashes = np.zeros((self.h, self.w), dtype=np.float32)

        for cell in self.cells:
            intensity = cell.current_intensity()
            if intensity <= 0:
                continue
            radar += _gaussian_blob(self.h, self.w, cell.row, cell.col, cell.radius, intensity)

            local_cape = self.cape_field[int(np.clip(cell.row, 0, self.h - 1)), int(np.clip(cell.col, 0, self.w - 1))]
            lightning_threshold = 0.75 - 0.35 * local_cape  # more unstable -> easier to spark
            if intensity > lightning_threshold:
                flash_prob = np.clip((intensity - lightning_threshold) * 3.0, 0, 0.9)
                if self.rng.random() < flash_prob:
                    jitter_r = cell.row + self.rng.normal(0, 1.0)
                    jitter_c = cell.col + self.rng.normal(0, 1.0)
                    new_flashes += _gaussian_blob(self.h, self.w, jitter_r, jitter_c, radius=1.2, amplitude=1.0)

        radar = np.clip(radar, 0, 1)

        # Lightning density: decay previous activity, add new flashes.
        self.lightning_density = np.clip(self.lightning_density * 0.65 + new_flashes, 0, 1).astype(np.float32)

        # Satellite IR: colder (lower value) over a smoothed/wider version of
        # the radar field (cloud shield extends beyond the radar core).
        cloud_shield = gaussian_filter(radar, sigma=3.0)
        satellite_ir = np.clip(1.0 - 0.85 * cloud_shield, 0, 1).astype(np.float32)

        return {
            "radar_dbz": radar.astype(np.float32),
            "satellite_ir_k": satellite_ir,
            "lightning_density": self.lightning_density.copy(),
            "cape": self.cape_field.copy(),
            "shear": self.shear_field.copy(),
        }

    def run(self, n_steps: int) -> list[dict[str, np.ndarray]]:
        return [self.step() for _ in range(n_steps)]


def stack_channels(frame: dict[str, np.ndarray]) -> np.ndarray:
    """dict[channel] -> (C, H, W) float32 array in the fixed CHANNELS order."""
    return np.stack([frame[c] for c in CHANNELS], axis=0).astype(np.float32)
