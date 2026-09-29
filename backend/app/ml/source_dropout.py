"""Source-dropout augmentation and the explicit per-source availability mask
— the concrete implementation of the "radar-optional by design" claim.

Why this file exists: the pitch deck and idea write-up describe an
"availability mask" carried alongside every input tensor, and a
"source-dropout" training scheme that randomly hides whole input channels
so the model learns to degrade gracefully instead of collapsing when radar
(or any other source) is missing. Neither existed anywhere in this codebase
before this file — `synthetic_generator.py` always emits a full, complete
five-channel frame, and nothing ever masked or dropped a channel. That is a
real, honest gap between the pitch and the prototype, not a difference in
wording.

This module closes it as a small, self-contained, OPT-IN utility rather
than by editing `synthetic_generator.py`, `dataset.py`, or `train.py`
in place: those files are exercised by the existing, passing test suite and
by the working demo, and changing their default behaviour is exactly the
kind of change the project's own documentation (see docs/technical_flow.md's
"frozen interface" note and NEXT_STEPS.md's running list of what's verified
vs. not) says to avoid without a clear reason. Wiring this into the main
training loop by default is a good next step — see the `--source-dropout`
flag on `backend/app/ml/train.py`'s CLI is NOT added by this change on
purpose — but that is a deliberate scope decision for a real training run,
not something to flip silently inside a hackathon submission pass.

Run `python -m app.ml.source_dropout` for a one-command, no-arguments demo
that prints a before/after example on real synthetic data.
"""
from __future__ import annotations

import numpy as np

from app.core.config import CHANNELS


def full_availability_mask(n_channels: int = len(CHANNELS)) -> np.ndarray:
    """The trivial "everything is present" mask — shape (n_channels,), all 1s.
    This is what every frame in this repo has implicitly assumed until now."""
    return np.ones(n_channels, dtype=np.float32)


def apply_source_dropout(
    channel_stack: np.ndarray,
    drop_prob: float = 0.3,
    min_available: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly zeroes out whole channels of `channel_stack` to simulate a
    missing data source (e.g. radar offline), and returns the masked stack
    alongside an explicit availability mask the model can condition on.

    channel_stack: (C, H, W) or (T, C, H, W) float32 array in CHANNELS order.
    drop_prob: independent probability each channel is dropped this call.
    min_available: never drop channels if fewer than this many would remain
        available afterward — a real deployment is never missing *every*
        source simultaneously, and training on that degenerate case would
        just teach the model to output noise.
    Returns: (masked_stack, availability_mask) where availability_mask has
        shape (C,) — 1.0 where the channel survived, 0.0 where it was
        dropped. For a (T, C, H, W) input, the SAME mask is applied across
        every time step, matching how a real source outage behaves (a
        radar that's down stays down for the whole window, it doesn't
        flicker frame to frame).
    """
    rng = rng or np.random.default_rng()
    channel_stack = np.asarray(channel_stack, dtype=np.float32)
    is_sequence = channel_stack.ndim == 4  # (T, C, H, W)
    n_channels = channel_stack.shape[1] if is_sequence else channel_stack.shape[0]

    mask = (rng.random(n_channels) >= drop_prob).astype(np.float32)
    if int(mask.sum()) < min_available:
        # Re-roll which channels survive rather than giving up on dropout
        # entirely, so min_available is a floor, not an excuse to skip
        # augmentation on this example.
        keep_idx = rng.choice(n_channels, size=min_available, replace=False)
        mask = np.zeros(n_channels, dtype=np.float32)
        mask[keep_idx] = 1.0

    if is_sequence:
        broadcast_mask = mask[None, :, None, None]
    else:
        broadcast_mask = mask[:, None, None]
    masked_stack = channel_stack * broadcast_mask
    return masked_stack, mask


def _demo() -> None:
    from app.core.config import GRID_H, GRID_W
    from app.data.synthetic_generator import StormGenerator, stack_channels

    gen = StormGenerator(rng_seed=7)
    frame = stack_channels(gen.step())  # (C, H, W), real synthetic data
    print(f"Channels (fixed order): {CHANNELS}")
    print(f"Frame shape: {frame.shape}, all channels present: {full_availability_mask()}")

    masked, mask = apply_source_dropout(frame, drop_prob=0.4, rng=np.random.default_rng(1))
    dropped = [c for c, m in zip(CHANNELS, mask) if m == 0.0]
    kept = [c for c, m in zip(CHANNELS, mask) if m == 1.0]
    print(f"After source dropout -> availability_mask={mask}")
    print(f"  dropped (simulated missing sources): {dropped or '(none this draw)'}")
    print(f"  kept: {kept}")
    for name, before, after, kept in zip(CHANNELS, frame, masked, mask):
        status = "kept" if kept == 1.0 else "ZEROED OUT (simulated missing source)"
        print(f"  {name}: mean before={before.mean():.4f}, mean after={after.mean():.4f} [{status}]")


if __name__ == "__main__":
    _demo()
