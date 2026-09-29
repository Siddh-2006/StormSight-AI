"""Virtual Radar Fill — reconstructs synthetic radar reflectivity from
satellite + lightning + NWP data alone, for when real Doppler radar is
missing or offline.

Why this file exists: this is the single most-emphasised idea in the pitch
deck and idea write-up ("radar-optional by design... Virtual Radar Fill...
GREMLIN-style"), and until now there was no code for it anywhere in this
repository — the ConvLSTM in model.py takes all five channels as a given
and has never been asked to produce radar from the other four. This module
is a compact U-Net (Ronneberger et al. 2015 — encoder/decoder with skip
connections, which is what lets it reconstruct sharp spatial detail rather
than a blurry average) doing the same image-to-image translation task
Hilburn et al.'s GREMLIN (JAMC 2021) does: satellite radiances (+ lightning,
in our case) in, radar-like reflectivity out — plus a second output channel
for the model's own per-pixel confidence, since a reconstructed estimate
should never be presented as equal to a real radar reading (see
docs/design.md's honesty conventions, carried through here).

Deliberately kept separate from `model.py`'s ConvLSTMNowcaster and NOT
wired into `infer.py` / the `/api/nowcast` endpoint: that endpoint's
contract is frozen (docs/technical_flow.md) and always assumes radar is
present in the input stack today. Training and trying this model is its own
one-command entry point — see `scripts/train_virtual_radar_fill.py` — so it
can be exercised and demoed on its own without touching the main, already-
tested nowcasting pipeline.
"""
from __future__ import annotations

import torch
from torch import nn

from app.core.config import CHANNELS

# Every channel except radar_dbz is a legitimate input; radar_dbz is what
# we're trying to reconstruct, so it must never leak into the input stack.
VRF_INPUT_CHANNELS = [c for c in CHANNELS if c != "radar_dbz"]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VirtualRadarFillUNet(nn.Module):
    """Small 2-level U-Net. Input (B, len(VRF_INPUT_CHANNELS), H, W) ->
    output (B, 2, H, W): channel 0 = reconstructed radar_dbz in [0,1],
    channel 1 = per-pixel confidence in [0,1] (1 = trust this pixel)."""

    def __init__(self, in_channels: int = len(VRF_INPUT_CHANNELS), base_ch: int = 16):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)  # *4 in: concat with enc2 skip
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)  # *2 in: concat with enc1 skip

        self.out_head = nn.Conv2d(base_ch, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W). Returns (B, 2, H, W), both channels sigmoided."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out_head(d1))

    @staticmethod
    def split_output(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """raw: (B, 2, H, W) from forward(). Returns (reconstructed_radar,
        confidence), each (B, H, W)."""
        return raw[:, 0], raw[:, 1]


def stack_vrf_inputs(frame: dict[str, "np.ndarray"]) -> "np.ndarray":  # noqa: F821 - numpy imported lazily below
    """dict[channel] -> (len(VRF_INPUT_CHANNELS), H, W) float32, matching
    VRF_INPUT_CHANNELS order. Mirrors synthetic_generator.stack_channels but
    deliberately excludes radar_dbz, the label this model predicts."""
    import numpy as np

    return np.stack([frame[c] for c in VRF_INPUT_CHANNELS], axis=0).astype("float32")
