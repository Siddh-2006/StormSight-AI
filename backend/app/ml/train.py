"""Quick training script for the ConvLSTM nowcaster on synthetic data.

Usage (from repo root, with venv activated):
    python -m app.ml.train --epochs 3 --episodes-per-epoch 120

This is intentionally a fast hackathon-scale training run (CPU-friendly,
minutes not hours) meant to prove the pipeline learns *something* sensible
(storms persist/move, lightning correlates with intense cells) rather than to
reach a fully converged operational model.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.core.config import CHECKPOINT_PATH, DEFAULT_HORIZON, HISTORY_LEN
from app.ml.dataset import SyntheticNowcastDataset
from app.ml.model import ConvLSTMNowcaster, focal_loss, weighted_mse_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train(
    epochs: int,
    episodes_per_epoch: int,
    batch_size: int,
    lr: float,
    horizon: int,
    hidden_channels: tuple[int, int] = (16, 32),
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # Small per-step tensors on this grid size see thread-launch overhead
        # dominate compute when torch defaults to all logical cores; a modest
        # thread count is consistently faster here. Irrelevant on GPU.
        torch.set_num_threads(min(4, torch.get_num_threads()))
    logger.info("Training on device=%s (threads=%d, hidden_channels=%s)", device, torch.get_num_threads(), hidden_channels)

    model = ConvLSTMNowcaster(in_channels=5, horizon=horizon, hidden_channels=hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        dataset = SyntheticNowcastDataset(epoch_size=episodes_per_epoch, horizon=horizon, base_seed=epoch * 10_000)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        epoch_start = time.time()
        running_radar_loss, running_lightning_loss, n_batches = 0.0, 0.0, 0

        for input_seq, radar_target, lightning_target in loader:
            input_seq = input_seq.to(device)
            radar_target = radar_target.to(device)
            lightning_target = lightning_target.to(device)

            radar_pred, lightning_pred = model(input_seq)

            radar_loss = weighted_mse_loss(radar_pred, radar_target)
            lightning_loss = focal_loss(lightning_pred, lightning_target)
            loss = radar_loss + lightning_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_radar_loss += radar_loss.item()
            running_lightning_loss += lightning_loss.item()
            n_batches += 1

        elapsed = time.time() - epoch_start
        logger.info(
            "epoch %d/%d  radar_mse=%.4f  lightning_focal=%.4f  (%.1fs, %d batches)",
            epoch + 1,
            epochs,
            running_radar_loss / max(n_batches, 1),
            running_lightning_loss / max(n_batches, 1),
            elapsed,
            n_batches,
        )

    ckpt_path = Path(CHECKPOINT_PATH)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "history_len": HISTORY_LEN,
            "horizon": horizon,
            "hidden_channels": list(hidden_channels),
        },
        ckpt_path,
    )
    logger.info("Saved checkpoint to %s", ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--episodes-per-epoch", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--hidden-channels",
        type=str,
        default="16,32",
        help="Comma-separated encoder/decoder hidden channel widths, e.g. '32,64' for a bigger model on more capable hardware (HPC).",
    )
    args = parser.parse_args()
    hidden_channels = tuple(int(x) for x in args.hidden_channels.split(","))
    train(args.epochs, args.episodes_per_epoch, args.batch_size, args.lr, args.horizon, hidden_channels)
