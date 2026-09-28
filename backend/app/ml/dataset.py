"""On-the-fly synthetic training data. Since the generator is procedural and
cheap, we don't materialize a fixed dataset on disk — each __getitem__ draws a
fresh random storm episode. This gives effectively unlimited training
variety, which matters because our only data source is simulated."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from app.core.config import HISTORY_LEN, DEFAULT_HORIZON, LIGHTNING_OCCURRENCE_THRESHOLD
from app.data.synthetic_generator import StormGenerator, stack_channels


class SyntheticNowcastDataset(Dataset):
    def __init__(
        self,
        epoch_size: int = 200,
        history_len: int = HISTORY_LEN,
        horizon: int = DEFAULT_HORIZON,
        base_seed: int = 0,
    ):
        self.epoch_size = epoch_size
        self.history_len = history_len
        self.horizon = horizon
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int):
        seed = self.base_seed + idx
        gen = StormGenerator(rng_seed=seed)
        gen.reset(seed=seed, num_cells=int(np.random.default_rng(seed).integers(1, 4)))
        frames = gen.run(self.history_len + self.horizon)
        stacked = [stack_channels(f) for f in frames]  # each (C, H, W)

        input_seq = np.stack(stacked[: self.history_len], axis=0)  # (T_in, C, H, W)
        future = stacked[self.history_len :]
        radar_target = np.stack([f[0] for f in future], axis=0)          # (T_out, H, W) channel 0 = radar
        lightning_raw = np.stack([f[2] for f in future], axis=0)         # channel 2 = lightning_density
        lightning_target = (lightning_raw > LIGHTNING_OCCURRENCE_THRESHOLD).astype(np.float32)

        return (
            torch.from_numpy(input_seq).float(),
            torch.from_numpy(radar_target).float(),
            torch.from_numpy(lightning_target).float(),
        )
