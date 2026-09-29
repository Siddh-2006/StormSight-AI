"""Inference engine used by the API layer. Wraps both the ConvLSTM model and
the optical-flow baseline behind one interface so routes_nowcast.py doesn't
care which is active."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from app.core.config import CHECKPOINT_PATH, DEFAULT_HORIZON, HISTORY_LEN
from app.ml import baseline
from app.ml.model import ConvLSTMNowcaster

logger = logging.getLogger(__name__)


class NowcastEngine:
    def __init__(self, horizon: int = DEFAULT_HORIZON):
        self.horizon = horizon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_channels = self._checkpoint_hidden_channels()
        self.model = ConvLSTMNowcaster(in_channels=5, horizon=horizon, hidden_channels=hidden_channels).to(self.device)
        self.model.eval()
        self._load_checkpoint()

    @staticmethod
    def _checkpoint_hidden_channels() -> tuple[int, int]:
        """Read hidden_channels from the checkpoint if present, so a bigger
        model trained elsewhere (e.g. on HPC, see scripts/train_hpc.sbatch)
        loads correctly without code changes here."""
        ckpt_path = Path(CHECKPOINT_PATH)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "hidden_channels" in ckpt:
                return tuple(ckpt["hidden_channels"])
        return (16, 32)

    def _load_checkpoint(self) -> None:
        ckpt_path = Path(CHECKPOINT_PATH)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            logger.info("Loaded ConvLSTM checkpoint from %s", ckpt_path)
        else:
            logger.warning(
                "No checkpoint found at %s — serving an UNTRAINED model. "
                "Run `python -m app.ml.train` first for meaningful nowcasts.",
                ckpt_path,
            )

    @torch.no_grad()
    def predict_ml(self, history_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """history_stack: (T_in, C, H, W) float32 in [0,1].
        Returns (radar_forecast, lightning_probability), each (horizon, H, W)."""
        x = torch.from_numpy(history_stack).float().unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        radar_pred, lightning_pred = self.model(x)
        return (
            radar_pred.squeeze(0).cpu().numpy(),
            lightning_pred.squeeze(0).cpu().numpy(),
        )

    def predict_baseline(self, history_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Optical-flow extrapolation of the radar channel; naive thresholded
        lightning proxy derived from it (see baseline.baseline_lightning_from_radar)."""
        radar_history = history_stack[:, 0, :, :]  # (T_in, H, W)
        radar_forecast = baseline.advect_frames(radar_history, self.horizon)
        lightning_forecast = baseline.baseline_lightning_from_radar(radar_forecast)
        return radar_forecast, lightning_forecast

    def predict(self, history_stack: np.ndarray, model: str = "convlstm") -> tuple[np.ndarray, np.ndarray]:
        if model == "baseline":
            return self.predict_baseline(history_stack)
        return self.predict_ml(history_stack)
