"""HPC training entrypoint for the ConvLSTM nowcaster. Standalone (no
FastAPI/backend dependency) — everything it needs lives in hpc_train/model/.

Usage (from hpc_train/model/):
    python train.py --data-source mixed --epochs 60 --episodes-per-epoch 500 \
        --batch-size 32 --hidden-channels 32,64

Logs to both console and a timestamped file under hpc_train/logs/ (the
--log-dir default), so a SLURM job's progress survives after the job log
truncates/rotates or you want to review it separately from stdout.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CHECKPOINT_PATH, DEFAULT_HORIZON, HISTORY_LEN, PROCESSED_SEVIR_DIR
from dataset import MixedDataset, RealEventDataset, SyntheticNowcastDataset
from model import ConvLSTMNowcaster, focal_loss, weighted_mse_loss

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )
    logger.info("Logging to %s", log_file)


def build_dataset(data_source: str, episodes_per_epoch: int, horizon: int, base_seed: int):
    if data_source == "synthetic":
        return SyntheticNowcastDataset(epoch_size=episodes_per_epoch, horizon=horizon, base_seed=base_seed)
    if data_source == "real":
        return RealEventDataset(processed_dir=PROCESSED_SEVIR_DIR, horizon=horizon)
    if data_source == "mixed":
        real = RealEventDataset(processed_dir=PROCESSED_SEVIR_DIR, horizon=horizon)
        synthetic = SyntheticNowcastDataset(epoch_size=episodes_per_epoch, horizon=horizon, base_seed=base_seed)
        return MixedDataset(real, synthetic)
    raise ValueError(f"unknown --data-source {data_source!r}")


def train(
    data_source: str,
    epochs: int,
    episodes_per_epoch: int,
    batch_size: int,
    lr: float,
    horizon: int,
    hidden_channels: tuple[int, int],
    num_workers: int,
    checkpoint_path: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    logger.info(
        "device=%s (%s) hidden_channels=%s data_source=%s",
        device,
        torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        hidden_channels,
        data_source,
    )

    model = ConvLSTMNowcaster(in_channels=5, horizon=horizon, hidden_channels=hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d (%.2fM)", n_params, n_params / 1e6)

    for epoch in range(epochs):
        dataset = build_dataset(data_source, episodes_per_epoch, horizon, base_seed=epoch * 10_000)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

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
            "epoch %d/%d  radar_mse=%.4f  lightning_focal=%.4f  (%.1fs, %d batches, %d samples)",
            epoch + 1,
            epochs,
            running_radar_loss / max(n_batches, 1),
            running_lightning_loss / max(n_batches, 1),
            elapsed,
            n_batches,
            len(dataset),
        )

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "history_len": HISTORY_LEN,
                    "horizon": horizon,
                    "hidden_channels": list(hidden_channels),
                    "epoch": epoch + 1,
                    "data_source": data_source,
                },
                checkpoint_path,
            )
            logger.info("Saved checkpoint (epoch %d) to %s", epoch + 1, checkpoint_path)

    logger.info("Training complete. Final checkpoint: %s", checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", choices=["synthetic", "real", "mixed"], default="mixed")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--episodes-per-epoch", type=int, default=500, help="Synthetic episodes per epoch (ignored for --data-source real).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--hidden-channels", type=str, default="32,64")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-path", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--log-dir", type=str, default="../logs")
    args = parser.parse_args()

    setup_logging(Path(args.log_dir))
    hidden_channels = tuple(int(x) for x in args.hidden_channels.split(","))
    train(
        args.data_source,
        args.epochs,
        args.episodes_per_epoch,
        args.batch_size,
        args.lr,
        args.horizon,
        hidden_channels,
        args.num_workers,
        Path(args.checkpoint_path),
    )
