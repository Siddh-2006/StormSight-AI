"""Correctness check for the nowcast_cpp C++ optical-flow baseline.

This is a plain script (run with `python test_advect.py`), not a pytest
suite, per the task brief -- though it happens to also be pytest-collectable
since the check is expressed as a single `test_...` function with asserts.

Method: generate a synthetic sequence of frames (pure NumPy, no dependency
on the rest of the repo) containing a single 2D Gaussian "blob" translating
at a known, constant pixel velocity. Feed the sequence to
`nowcast_cpp.advect_frames`, compute the intensity-weighted centroid of each
predicted frame, and check it lands close (within a few pixels) to the
analytically expected position: last_observed_position + step * velocity.

This directly exercises the frozen contract end-to-end:
    advect_frames(frames: (T,H,W) float32, n_steps: int) -> (n_steps,H,W) float32
which internally calls estimate_flow() on the last two frames and then
semi-Lagrangian-advects with bilinear interpolation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import nowcast_cpp

H, W = 64, 64
SIGMA = 4.0
N_INPUT_FRAMES = 4
N_STEPS = 4
# Known constant velocity, in pixels/frame: (row displacement, col displacement).
VELOCITY = (1.5, 2.0)
START_CENTER = (24.0, 18.0)  # (row, col) at the first synthetic frame (t=0)


def make_gaussian_frame(center_row, center_col, height=H, width=W, sigma=SIGMA):
    rows = np.arange(height, dtype=np.float32).reshape(-1, 1)
    cols = np.arange(width, dtype=np.float32).reshape(1, -1)
    dr = rows - center_row
    dc = cols - center_col
    frame = np.exp(-(dr * dr + dc * dc) / (2.0 * sigma * sigma))
    return frame.astype(np.float32)


def weighted_centroid(frame):
    """Intensity-weighted centroid (row, col) of a 2D field."""
    total = frame.sum()
    rows = np.arange(frame.shape[0], dtype=np.float64).reshape(-1, 1)
    cols = np.arange(frame.shape[1], dtype=np.float64).reshape(1, -1)
    centroid_row = float((frame * rows).sum() / total)
    centroid_col = float((frame * cols).sum() / total)
    return centroid_row, centroid_col


def build_sequence():
    frames = []
    for t in range(N_INPUT_FRAMES):
        r = START_CENTER[0] + t * VELOCITY[0]
        c = START_CENTER[1] + t * VELOCITY[1]
        frames.append(make_gaussian_frame(r, c))
    return np.stack(frames, axis=0).astype(np.float32)


def test_advect_matches_known_constant_velocity():
    frames = build_sequence()
    assert frames.shape == (N_INPUT_FRAMES, H, W)
    assert frames.dtype == np.float32

    last_center = (
        START_CENTER[0] + (N_INPUT_FRAMES - 1) * VELOCITY[0],
        START_CENTER[1] + (N_INPUT_FRAMES - 1) * VELOCITY[1],
    )

    predicted = nowcast_cpp.advect_frames(frames, N_STEPS)

    assert predicted.shape == (N_STEPS, H, W), predicted.shape
    assert predicted.dtype == np.float32

    max_row_err = 0.0
    max_col_err = 0.0
    for step in range(N_STEPS):
        n = step + 1
        expected_row = last_center[0] + n * VELOCITY[0]
        expected_col = last_center[1] + n * VELOCITY[1]

        got_row, got_col = weighted_centroid(predicted[step])

        row_err = abs(got_row - expected_row)
        col_err = abs(got_col - expected_col)
        max_row_err = max(max_row_err, row_err)
        max_col_err = max(max_col_err, col_err)

        print(
            f"step {n}: expected center=({expected_row:.2f}, {expected_col:.2f}) "
            f"got=({got_row:.2f}, {got_col:.2f}) "
            f"err=({row_err:.2f}, {col_err:.2f})"
        )

    tolerance_px = 3.0
    assert max_row_err < tolerance_px, f"row centroid error {max_row_err:.2f}px exceeds {tolerance_px}px"
    assert max_col_err < tolerance_px, f"col centroid error {max_col_err:.2f}px exceeds {tolerance_px}px"

    print(f"PASS: max row error {max_row_err:.3f}px, max col error {max_col_err:.3f}px "
          f"(tolerance {tolerance_px}px)")


def test_estimate_flow_shapes_and_direction():
    """Sanity check on estimate_flow in isolation, per the frozen contract:
    estimate_flow(frame_a, frame_b) -> (u, v), each (H, W) float32."""
    frame_a = make_gaussian_frame(*START_CENTER)
    r2 = START_CENTER[0] + VELOCITY[0]
    c2 = START_CENTER[1] + VELOCITY[1]
    frame_b = make_gaussian_frame(r2, c2)

    u, v = nowcast_cpp.estimate_flow(frame_a, frame_b)

    assert u.shape == (H, W)
    assert v.shape == (H, W)
    assert u.dtype == np.float32
    assert v.dtype == np.float32

    # Sample flow near the blob center -- u is column (horizontal)
    # displacement, v is row (vertical) displacement, matching the
    # documented convention used by advect_frames' semi-Lagrangian step.
    r0, c0 = int(round(START_CENTER[0])), int(round(START_CENTER[1]))
    window = 6
    local_u = u[r0 - window:r0 + window, c0 - window:c0 + window].mean()
    local_v = v[r0 - window:r0 + window, c0 - window:c0 + window].mean()
    print(f"estimate_flow near blob: mean local u={local_u:.3f} (expect ~{VELOCITY[1]}), "
          f"mean local v={local_v:.3f} (expect ~{VELOCITY[0]})")

    assert local_u > 0.3, "expected a positive (rightward) column flow near the moving blob"
    assert local_v > 0.2, "expected a positive (downward) row flow near the moving blob"


if __name__ == "__main__":
    test_estimate_flow_shapes_and_direction()
    test_advect_matches_known_constant_velocity()
    print("ALL TESTS PASSED")
