"""Compact ConvLSTM encoder-forecaster, SmaAt-UNet-family lite (depthwise-
separable convs) — see docs/research_notes.md section 1 for why this
architecture family was chosen over heavier options (DGMR/NowcastNet/MetNet)
for a hackathon compute/time budget.

Input:  (B, T_in, C, H, W)  — T_in = HISTORY_LEN frames of all CHANNELS.
Output: two decoder heads, each (B, T_out, H, W):
  - radar_forecast: regression in [0, 1] (sigmoid)
  - lightning_forecast: probability in [0, 1] (sigmoid)
"""
from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ConvLSTMCell(nn.Module):
    """Standard ConvLSTM cell (Shi et al., NeurIPS 2015) using a depthwise-
    separable conv for the gate computation to keep the parameter count small."""

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.gates = DepthwiseSeparableConv(in_ch + hidden_ch, 4 * hidden_ch, kernel_size)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch: int, h: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch, self.hidden_ch, h, w)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)


class ConvLSTMNowcaster(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: tuple[int, int] = (16, 32),
        horizon: int = 12,
    ):
        super().__init__()
        self.horizon = horizon
        self.in_channels = in_channels

        self.encoder1 = ConvLSTMCell(in_channels, hidden_channels[0])
        self.encoder2 = ConvLSTMCell(hidden_channels[0], hidden_channels[1])

        # Decoder input = zeros + a scalar lead-time positional channel,
        # concatenated to in_channels width so the decoder cell can be reused.
        self.decoder1 = ConvLSTMCell(in_channels, hidden_channels[0])
        self.decoder2 = ConvLSTMCell(hidden_channels[0], hidden_channels[1])

        self.radar_head = nn.Sequential(
            DepthwiseSeparableConv(hidden_channels[1], 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )
        self.lightning_head = nn.Sequential(
            DepthwiseSeparableConv(hidden_channels[1], 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T_in, C, H, W)."""
        b, t_in, c, h, w = x.shape
        device = x.device

        h1, c1 = self.encoder1.init_state(b, h, w, device)
        h2, c2 = self.encoder2.init_state(b, h, w, device)
        for t in range(t_in):
            h1, c1 = self.encoder1(x[:, t], (h1, c1))
            h2, c2 = self.encoder2(h1, (h2, c2))

        # Hand encoder's final states to the decoder stack.
        dh1, dc1 = h1, c1
        dh2, dc2 = h2, c2

        radar_out, lightning_out = [], []
        zero_input = torch.zeros(b, c, h, w, device=device)
        for step in range(self.horizon):
            lead_channel = torch.full((b, 1, h, w), step / max(self.horizon - 1, 1), device=device)
            dec_input = zero_input.clone()
            dec_input[:, -1:, :, :] = lead_channel  # reuse last channel slot as positional cue
            dh1, dc1 = self.decoder1(dec_input, (dh1, dc1))
            dh2, dc2 = self.decoder2(dh1, (dh2, dc2))
            radar_out.append(torch.sigmoid(self.radar_head(dh2)))
            lightning_out.append(torch.sigmoid(self.lightning_head(dh2)))

        radar_forecast = torch.cat(radar_out, dim=1)          # (B, horizon, H, W)
        lightning_forecast = torch.cat(lightning_out, dim=1)  # (B, horizon, H, W)
        return radar_forecast, lightning_forecast


def focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.85) -> torch.Tensor:
    """Binary focal loss — lightning occurrence is a rare-event class, plain
    BCE under-predicts it (see docs/design.md section 2.2)."""
    eps = 1e-7
    pred = pred.clamp(eps, 1 - eps)
    pt = torch.where(target == 1, pred, 1 - pred)
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    loss = -alpha_t * (1 - pt) ** gamma * torch.log(pt)
    return loss.mean()
