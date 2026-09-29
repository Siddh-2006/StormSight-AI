"""Standalone training + demo for Virtual Radar Fill (see
backend/app/ml/virtual_radar_fill.py for what this is and why it's separate
from the main nowcasting pipeline).

One-click usage (from repo root, venv activated):
    .venv/Scripts/python scripts/train_virtual_radar_fill.py

Defaults are deliberately small (2 epochs, 60 synthetic episodes) so this
finishes in well under a minute on a laptop CPU and is genuinely a "run one
command, see it work" demo, not a multi-hour training job — pass
--epochs/--episodes-per-epoch for a more serious run. Self-supervised: every
synthetic frame already has a real radar_dbz channel, so there's no separate
labelled dataset to source — the label is simply the channel this model is
NOT allowed to see as input (see VRF_INPUT_CHANNELS).

Saves a checkpoint to models/checkpoints/virtual_radar_fill.pt (separate
from convlstm.pt — this is a different model for a different job) and, at
the end, prints a concrete before/after: given only satellite + lightning +
CAPE + shear for a fresh, unseen synthetic scene, how close is the
reconstructed radar to the real (withheld) radar, and what confidence does
the model report.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np
import torch

torch.set_num_threads(min(4, torch.get_num_threads()))  # see backend/app/ml/train.py

from app.data.synthetic_generator import StormGenerator
from app.ml.virtual_radar_fill import VirtualRadarFillUNet, stack_vrf_inputs

VRF_CHECKPOINT_PATH = REPO_ROOT / "models" / "checkpoints" / "virtual_radar_fill.pt"


def make_batch(batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Draws `batch_size` independent single-frame synthetic scenes (each
    from its own freshly-seeded generator, so a batch spans varied storm
    layouts rather than one storm's timeline) and returns
    (vrf_inputs, radar_target), shapes (B, 4, H, W) and (B, H, W)."""
    inputs, targets = [], []
    for i in range(batch_size):
        gen = StormGenerator(rng_seed=seed + i)
        for _ in range(np.random.randint(1, 6)):  # a few warm-up steps so storms aren't all brand new
            frame = gen.step()
        inputs.append(stack_vrf_inputs(frame))
        targets.append(frame["radar_dbz"])
    return (
        torch.from_numpy(np.stack(inputs, axis=0)).float(),
        torch.from_numpy(np.stack(targets, axis=0)).float(),
    )


def reconstruction_loss(pred_radar: torch.Tensor, pred_confidence: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on the reconstruction, plus a light penalty encouraging confidence
    to track actual pixel-wise accuracy (low error -> should report high
    confidence, high error -> should report low confidence) rather than
    letting the confidence head collapse to a constant."""
    per_pixel_error = (pred_radar - target) ** 2
    recon_loss = per_pixel_error.mean()
    implied_confidence_target = torch.exp(-5.0 * per_pixel_error.detach())  # low error -> target near 1
    confidence_loss = ((pred_confidence - implied_confidence_target) ** 2).mean()
    return recon_loss + 0.3 * confidence_loss


def train(epochs: int, episodes_per_epoch: int, batch_size: int, lr: float) -> VirtualRadarFillUNet:
    model = VirtualRadarFillUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss, n_batches = 0.0, 0
        n_batches_this_epoch = max(1, episodes_per_epoch // batch_size)
        for b in range(n_batches_this_epoch):
            x, target = make_batch(batch_size, seed=epoch * 10_000 + b * 100)
            raw = model(x)
            pred_radar, pred_confidence = VirtualRadarFillUNet.split_output(raw)
            loss = reconstruction_loss(pred_radar, pred_confidence, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        print(
            f"[virtual-radar-fill] epoch {epoch + 1}/{epochs}  "
            f"loss={running_loss / max(n_batches, 1):.4f}  ({time.time() - epoch_start:.1f}s)"
        )

    VRF_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, VRF_CHECKPOINT_PATH)
    print(f"[virtual-radar-fill] saved checkpoint to {VRF_CHECKPOINT_PATH}")
    return model


@torch.no_grad()
def demo(model: VirtualRadarFillUNet) -> None:
    """Fresh, unseen scene: reconstruct radar from satellite+lightning+NWP
    only, compare to the real (withheld) radar, and report confidence."""
    model.eval()
    x, target = make_batch(batch_size=1, seed=999_999)
    raw = model(x)
    pred_radar, pred_confidence = VirtualRadarFillUNet.split_output(raw)

    mae = (pred_radar[0] - target[0]).abs().mean().item()
    mean_confidence = pred_confidence[0].mean().item()
    print("\n[virtual-radar-fill] demo on an unseen scene (radar withheld from the model):")
    print(f"  mean absolute reconstruction error vs. real (withheld) radar: {mae:.4f}  (scale: 0-1 normalized dBZ)")
    print(f"  mean self-reported confidence: {mean_confidence:.4f}")
    print("  -> this is the number a forecaster would see instead of a real radar reading")
    print("     whenever DWR coverage is down; it is deliberately never presented as equal")
    print("     to a genuine radar observation (see routes_alerts_explain.py's honesty note).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--episodes-per-epoch", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    trained = train(args.epochs, args.episodes_per_epoch, args.batch_size, args.lr)
    demo(trained)
