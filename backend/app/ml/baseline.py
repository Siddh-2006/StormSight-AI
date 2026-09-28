"""Classical optical-flow extrapolation baseline — the same operational
principle pySTEPS uses (see docs/research_notes.md section 1). This is the
"does the ML model actually add value" comparison point.

Tries to use the compiled C++ module (`nowcast_cpp`, built from
cpp_module/, see docs/technical_flow.md section 4) for speed/demonstration
of the C++ implementation; if it isn't built/importable, falls back to an
OpenCV-based pure-Python implementation with the identical function
signatures, so the rest of the system works either way.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import nowcast_cpp  # type: ignore

    _HAS_CPP = True
    logger.info("Using compiled nowcast_cpp module for optical-flow baseline.")
except ImportError:
    _HAS_CPP = False
    logger.info("nowcast_cpp not built/importable — using OpenCV/NumPy fallback baseline.")


def _estimate_flow_py(frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = (frame_a * 255).astype(np.uint8)
    b = (frame_b * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        a, b, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)
    return u, v


def _advect_frames_py(frames: np.ndarray, n_steps: int) -> np.ndarray:
    h, w = frames.shape[1], frames.shape[2]
    u, v = _estimate_flow_py(frames[-2], frames[-1])
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    current = frames[-1].copy()
    out = np.zeros((n_steps, h, w), dtype=np.float32)
    for step in range(n_steps):
        map_x = (grid_x + u * (step + 1)).astype(np.float32)
        map_y = (grid_y + v * (step + 1)).astype(np.float32)
        advected = cv2.remap(current, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out[step] = advected
    return out


def estimate_flow(frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if _HAS_CPP:
        return nowcast_cpp.estimate_flow(frame_a.astype(np.float32), frame_b.astype(np.float32))
    return _estimate_flow_py(frame_a, frame_b)


def advect_frames(frames: np.ndarray, n_steps: int) -> np.ndarray:
    frames = frames.astype(np.float32)
    if _HAS_CPP:
        return nowcast_cpp.advect_frames(frames, n_steps)
    return _advect_frames_py(frames, n_steps)


def baseline_lightning_from_radar(radar_forecast: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """The optical-flow baseline has no notion of lightning physics, so as a
    simple (deliberately naive) baseline lightning proxy we threshold the
    advected radar field — this is what the ML model's dual-head design is
    meant to beat (see docs/design.md section 2.2/2.3)."""
    return np.clip((radar_forecast - threshold) / (1 - threshold), 0, 1)
