// advect.hpp
//
// Pure C++ (no OpenCV, no Eigen) implementation of:
//   - dense optical-flow estimation between two frames (Horn-Schunck style
//     iterative differential method), and
//   - semi-Lagrangian advection of a frame forward in time using a constant
//     flow field, with bilinear interpolation.
//
// This is the numeric core behind the `nowcast_cpp` pybind11 module. It is
// deliberately dependency-free (plain std::vector<float> buffers) so it
// builds with nothing beyond a C++17 compiler.
//
// Frozen contract (see docs/technical_flow.md §4):
//   estimate_flow(frame_a, frame_b) -> (u, v)   each (H, W) float32
//   advect_frames(frames, n_steps)  -> (n_steps, H, W) float32
#pragma once

#include <cstddef>
#include <vector>

namespace nowcast {

// Simple row-major (H, W) float buffer view helpers are implemented with
// plain std::vector<float> of length H*W; index (r, c) -> r * W + c.

// Estimate dense optical flow (u = horizontal / column displacement,
// v = vertical / row displacement) describing motion from frame_a to
// frame_b, using an iterative Horn-Schunck-style differential method.
//
// frame_a, frame_b: row-major (H, W) buffers, values expected in [0, 1].
// u, v: output row-major (H, W) buffers (resized internally).
//
// alpha: smoothness regularization weight (Horn-Schunck lambda).
// iterations: number of Gauss-Seidel relaxation iterations.
void estimate_flow(const std::vector<float>& frame_a,
                    const std::vector<float>& frame_b,
                    std::size_t height,
                    std::size_t width,
                    std::vector<float>& u,
                    std::vector<float>& v,
                    float alpha = 0.01f,
                    int iterations = 200);

// Bilinear sample of a row-major (H, W) buffer at fractional coordinates
// (row, col). Out-of-bounds coordinates are clamped to the image border
// (replicate-edge behavior), which is the standard, stable choice for
// semi-Lagrangian advection near domain boundaries.
float bilinear_sample(const std::vector<float>& frame,
                       std::size_t height,
                       std::size_t width,
                       float row,
                       float col);

// Semi-Lagrangian advection: given the last frame and a constant flow field
// (u, v) estimated from the last two input frames, produce n_steps future
// frames by repeatedly backward-tracing along the flow and bilinearly
// resampling.
//
// last_frame: row-major (H, W) buffer (the most recent observed frame).
// u, v: row-major (H, W) flow fields (displacement per step, in pixels).
// n_steps: number of future steps to extrapolate.
// out: output buffer of size n_steps * H * W, frame t at
//      out[t * H * W : (t+1) * H * W].
void advect(const std::vector<float>& last_frame,
            const std::vector<float>& u,
            const std::vector<float>& v,
            std::size_t height,
            std::size_t width,
            int n_steps,
            std::vector<float>& out);

}  // namespace nowcast
