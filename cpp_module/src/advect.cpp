// advect.cpp
//
// From-scratch C++ implementation (no OpenCV/Eigen) of dense optical flow
// (Horn-Schunck) and semi-Lagrangian advection. See include/advect.hpp for
// the contract this implements.

#include "advect.hpp"

#include <algorithm>
#include <cmath>

namespace nowcast {

namespace {

inline std::size_t idx(std::size_t r, std::size_t c, std::size_t width) {
    return r * width + c;
}

// Clamp an index into [0, size-1].
inline std::size_t clamp_idx(long v, long size) {
    if (v < 0) return 0;
    if (v >= size) return static_cast<std::size_t>(size - 1);
    return static_cast<std::size_t>(v);
}

// Central-difference spatial gradients averaged over both frames (a common
// Horn-Schunck refinement over using frame_a alone), plus temporal
// difference. All buffers are row-major (H, W).
void compute_gradients(const std::vector<float>& frame_a,
                        const std::vector<float>& frame_b,
                        std::size_t height,
                        std::size_t width,
                        std::vector<float>& Ix,
                        std::vector<float>& Iy,
                        std::vector<float>& It) {
    const std::size_t n = height * width;
    Ix.assign(n, 0.0f);
    Iy.assign(n, 0.0f);
    It.assign(n, 0.0f);

    const long H = static_cast<long>(height);
    const long W = static_cast<long>(width);

    for (long r = 0; r < H; ++r) {
        for (long c = 0; c < W; ++c) {
            const std::size_t rp1 = clamp_idx(r + 1, H);
            const std::size_t rm1 = clamp_idx(r - 1, H);
            const std::size_t cp1 = clamp_idx(c + 1, W);
            const std::size_t cm1 = clamp_idx(c - 1, W);
            const std::size_t here = idx(static_cast<std::size_t>(r),
                                          static_cast<std::size_t>(c), width);

            // Central difference in x (columns) and y (rows), averaged
            // between the two frames for symmetric temporal accuracy.
            const float gx_a = (frame_a[idx(static_cast<std::size_t>(r), cp1, width)] -
                                 frame_a[idx(static_cast<std::size_t>(r), cm1, width)]) *
                                0.5f;
            const float gx_b = (frame_b[idx(static_cast<std::size_t>(r), cp1, width)] -
                                 frame_b[idx(static_cast<std::size_t>(r), cm1, width)]) *
                                0.5f;
            const float gy_a = (frame_a[idx(rp1, static_cast<std::size_t>(c), width)] -
                                 frame_a[idx(rm1, static_cast<std::size_t>(c), width)]) *
                                0.5f;
            const float gy_b = (frame_b[idx(rp1, static_cast<std::size_t>(c), width)] -
                                 frame_b[idx(rm1, static_cast<std::size_t>(c), width)]) *
                                0.5f;

            Ix[here] = 0.5f * (gx_a + gx_b);
            Iy[here] = 0.5f * (gy_a + gy_b);
            It[here] = frame_b[here] - frame_a[here];
        }
    }
}

}  // namespace

void estimate_flow(const std::vector<float>& frame_a,
                    const std::vector<float>& frame_b,
                    std::size_t height,
                    std::size_t width,
                    std::vector<float>& u,
                    std::vector<float>& v,
                    float alpha,
                    int iterations) {
    const std::size_t n = height * width;
    u.assign(n, 0.0f);
    v.assign(n, 0.0f);

    std::vector<float> Ix, Iy, It;
    compute_gradients(frame_a, frame_b, height, width, Ix, Iy, It);

    std::vector<float> u_next(n, 0.0f);
    std::vector<float> v_next(n, 0.0f);

    const long H = static_cast<long>(height);
    const long W = static_cast<long>(width);
    const float alpha2 = alpha * alpha;

    // Horn-Schunck iterative (Jacobi-style) relaxation:
    //   ubar, vbar = weighted local neighborhood average of u, v
    //   u = ubar - Ix * (Ix*ubar + Iy*vbar + It) / (alpha^2 + Ix^2 + Iy^2)
    //   v = vbar - Iy * (Ix*ubar + Iy*vbar + It) / (alpha^2 + Ix^2 + Iy^2)
    for (int iter = 0; iter < iterations; ++iter) {
        for (long r = 0; r < H; ++r) {
            for (long c = 0; c < W; ++c) {
                const std::size_t rp1 = clamp_idx(r + 1, H);
                const std::size_t rm1 = clamp_idx(r - 1, H);
                const std::size_t cp1 = clamp_idx(c + 1, W);
                const std::size_t cm1 = clamp_idx(c - 1, W);
                const std::size_t rr = static_cast<std::size_t>(r);
                const std::size_t cc = static_cast<std::size_t>(c);
                const std::size_t here = idx(rr, cc, width);

                // Standard Horn-Schunck 4-neighbor average (cross-shaped
                // stencil weight 1/6 for direct neighbors, plus diagonals at
                // 1/12 is the "classic" 12-neighborhood; the simpler 4
                // neighbor average is a common, adequate simplification for
                // a from-scratch implementation).
                const float ubar = 0.25f * (u[idx(rm1, cc, width)] + u[idx(rp1, cc, width)] +
                                             u[idx(rr, cm1, width)] + u[idx(rr, cp1, width)]);
                const float vbar = 0.25f * (v[idx(rm1, cc, width)] + v[idx(rp1, cc, width)] +
                                             v[idx(rr, cm1, width)] + v[idx(rr, cp1, width)]);

                const float ix = Ix[here];
                const float iy = Iy[here];
                const float it = It[here];

                const float numerator = ix * ubar + iy * vbar + it;
                const float denom = alpha2 + ix * ix + iy * iy;

                u_next[here] = ubar - ix * numerator / denom;
                v_next[here] = vbar - iy * numerator / denom;
            }
        }
        std::swap(u, u_next);
        std::swap(v, v_next);
    }
}

float bilinear_sample(const std::vector<float>& frame,
                       std::size_t height,
                       std::size_t width,
                       float row,
                       float col) {
    const float max_row = static_cast<float>(height) - 1.0f;
    const float max_col = static_cast<float>(width) - 1.0f;

    // Clamp to the image border (replicate-edge) so advection stays stable
    // when flow carries sample points outside the domain.
    row = std::clamp(row, 0.0f, max_row);
    col = std::clamp(col, 0.0f, max_col);

    const std::size_t r0 = static_cast<std::size_t>(std::floor(row));
    const std::size_t c0 = static_cast<std::size_t>(std::floor(col));
    const std::size_t r1 = std::min(r0 + 1, height - 1);
    const std::size_t c1 = std::min(c0 + 1, width - 1);

    const float fr = row - static_cast<float>(r0);
    const float fc = col - static_cast<float>(c0);

    const float v00 = frame[idx(r0, c0, width)];
    const float v01 = frame[idx(r0, c1, width)];
    const float v10 = frame[idx(r1, c0, width)];
    const float v11 = frame[idx(r1, c1, width)];

    const float top = v00 * (1.0f - fc) + v01 * fc;
    const float bottom = v10 * (1.0f - fc) + v11 * fc;
    return top * (1.0f - fr) + bottom * fr;
}

void advect(const std::vector<float>& last_frame,
            const std::vector<float>& u,
            const std::vector<float>& v,
            std::size_t height,
            std::size_t width,
            int n_steps,
            std::vector<float>& out) {
    const std::size_t n = height * width;
    out.assign(static_cast<std::size_t>(n_steps) * n, 0.0f);

    std::vector<float> current = last_frame;
    std::vector<float> next(n, 0.0f);

    for (int step = 0; step < n_steps; ++step) {
        for (std::size_t r = 0; r < height; ++r) {
            for (std::size_t c = 0; c < width; ++c) {
                const std::size_t here = idx(r, c, width);
                // Semi-Lagrangian: the value that will arrive at (r, c) at
                // the next step is the value currently located at the
                // upstream point (r, c) traced backward along the flow,
                // i.e. displaced by -1 step of (v, u) (v = row/vertical
                // displacement, u = col/horizontal displacement).
                const float src_row = static_cast<float>(r) - v[here];
                const float src_col = static_cast<float>(c) - u[here];
                next[here] = bilinear_sample(current, height, width, src_row, src_col);
            }
        }
        std::copy(next.begin(), next.end(), out.begin() + static_cast<std::size_t>(step) * n);
        std::swap(current, next);
    }
}

}  // namespace nowcast
