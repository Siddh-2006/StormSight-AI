"""Process-wide simulation state: the synthetic generator's rolling history
buffer and the inference engine, shared by all API routes.

Scoring note: `/api/nowcast` scores are computed by cloning the generator and
stepping the CLONE forward to reveal "true" future frames, without mutating
the real displayed timeline. This is an honest synthetic-replay verification
(same principle as verifying nowcasts against observed future radar in real
operations) — see docs/design.md section 2.3 and docs/technical_flow.md
section 3. It only works because everything here is simulated; a production
deployment would instead score against observations once they arrive.
"""
from __future__ import annotations

import copy
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np

from app.core.config import CHANNELS, HISTORY_LEN, TIME_STEP_MIN
from app.data.synthetic_generator import StormGenerator, stack_channels
from app.ml.infer import NowcastEngine


class SimulationState:
    def __init__(self) -> None:
        self.generator = StormGenerator(rng_seed=None)
        self.history: deque[dict[str, np.ndarray]] = deque(maxlen=HISTORY_LEN)
        self.engine = NowcastEngine()
        self.current_time = datetime.now(timezone.utc)
        self._warm_up()

    def _warm_up(self) -> None:
        for _ in range(HISTORY_LEN):
            self.history.append(self.generator.step())

    def reset(self, seed: int | None = None, num_cells: int = 2) -> None:
        self.generator.reset(seed=seed, num_cells=num_cells)
        self.history.clear()
        self.current_time = datetime.now(timezone.utc)
        self._warm_up()

    def step(self) -> dict[str, np.ndarray]:
        frame = self.generator.step()
        self.history.append(frame)
        self.current_time += timedelta(minutes=TIME_STEP_MIN)
        return frame

    def latest_frame(self) -> dict[str, np.ndarray]:
        return self.history[-1]

    def history_stack(self) -> np.ndarray:
        """(T_in, C, H, W) float32, oldest first."""
        return np.stack([stack_channels(f) for f in self.history], axis=0)

    def future_truth(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Clone the generator and step it forward `horizon` times to reveal
        true future radar + lightning-occurrence frames, without mutating the
        real timeline. Returns (radar_truth, lightning_density_truth), each
        (horizon, H, W)."""
        clone = copy.deepcopy(self.generator)
        radar_frames, lightning_frames = [], []
        for _ in range(horizon):
            f = clone.step()
            radar_frames.append(f["radar_dbz"])
            lightning_frames.append(f["lightning_density"])
        return np.stack(radar_frames, axis=0), np.stack(lightning_frames, axis=0)


state = SimulationState()
