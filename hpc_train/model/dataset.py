"""Training datasets for HPC: synthetic (procedural, unlimited), real
(preprocessed SEVIR events from data_prep/preprocess.py), or a mix of both.

Real event files are .npz with keys ["radar_dbz","satellite_ir_k",
"lightning_density","cape","shear"], each shape (PROJECT_SEQ_LEN, 64, 64)
float32 in [0,1] — see data_prep/preprocess.py for exactly how these were
derived and verified against live SEVIR data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import CHANNELS, DEFAULT_HORIZON, HISTORY_LEN, LIGHTNING_OCCURRENCE_THRESHOLD, PROCESSED_SEVIR_DIR
from synthetic_generator import StormGenerator, stack_channels


def _split_event(stacked_channels: dict[str, np.ndarray], history_len: int, horizon: int):
    """stacked_channels: {channel: (T, H, W)} with T >= history_len+horizon.
    Returns (input_seq (T_in,C,H,W), radar_target (T_out,H,W), lightning_target (T_out,H,W))."""
    full = np.stack([stacked_channels[c] for c in CHANNELS], axis=1)  # (T, C, H, W)
    input_seq = full[:history_len]
    future = full[history_len : history_len + horizon]
    radar_target = future[:, CHANNELS.index("radar_dbz")]
    lightning_raw = future[:, CHANNELS.index("lightning_density")]
    lightning_target = (lightning_raw > LIGHTNING_OCCURRENCE_THRESHOLD).astype(np.float32)
    return input_seq.astype(np.float32), radar_target.astype(np.float32), lightning_target.astype(np.float32)


class SyntheticNowcastDataset(Dataset):
    """Procedural synthetic episodes — see backend/app/data/synthetic_generator.py
    (copied here) for the storm-cell simulation. Unlimited variety; a fresh
    episode is generated per __getitem__ call."""

    def __init__(self, epoch_size: int = 200, history_len: int = HISTORY_LEN, horizon: int = DEFAULT_HORIZON, base_seed: int = 0):
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
        stacked = {c: np.stack([f[c] for f in frames], axis=0) for c in CHANNELS}
        input_seq, radar_target, lightning_target = _split_event(stacked, self.history_len, self.horizon)
        return torch.from_numpy(input_seq), torch.from_numpy(radar_target), torch.from_numpy(lightning_target)


class RealEventDataset(Dataset):
    """Loads preprocessed real SEVIR events (data_prep/preprocess.py output).
    Each .npz already spans HISTORY_LEN+DEFAULT_HORIZON steps at the
    project's 10-min cadence — see that script's PROJECT_SEQ_LEN."""

    def __init__(self, processed_dir: str = PROCESSED_SEVIR_DIR, history_len: int = HISTORY_LEN, horizon: int = DEFAULT_HORIZON):
        self.files = sorted(Path(processed_dir).glob("event_*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No processed event files found in {processed_dir}. "
                "Run data_prep/download_sevir.py then data_prep/preprocess.py first."
            )
        self.history_len = history_len
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx])
        stacked = {c: data[c] for c in CHANNELS}
        input_seq, radar_target, lightning_target = _split_event(stacked, self.history_len, self.horizon)
        return torch.from_numpy(input_seq), torch.from_numpy(radar_target), torch.from_numpy(lightning_target)


class MixedDataset(Dataset):
    """Concatenates a SyntheticNowcastDataset and a RealEventDataset so a
    single epoch trains on both — real data anchors the model to true storm
    physics, synthetic data supplies volume. Indexing: first len(real) items
    come from real data (repeated if synthetic_ratio makes the epoch longer
    than real alone), remainder from synthetic."""

    def __init__(self, real: RealEventDataset, synthetic: SyntheticNowcastDataset):
        self.real = real
        self.synthetic = synthetic

    def __len__(self) -> int:
        return len(self.real) + len(self.synthetic)

    def __getitem__(self, idx: int):
        if idx < len(self.real):
            return self.real[idx]
        return self.synthetic[idx - len(self.real)]
