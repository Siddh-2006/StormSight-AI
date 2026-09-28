// bindings.cpp
//
// pybind11 bindings exposing the C++ optical-flow baseline as the Python
// module `nowcast_cpp`, per the frozen contract in
// docs/technical_flow.md §4:
//
//   estimate_flow(frame_a, frame_b) -> (u, v)     each (H, W) float32
//   advect_frames(frames, n_steps)  -> (n_steps, H, W) float32

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <vector>

#include "advect.hpp"

namespace py = pybind11;

namespace {

// Copy a 2D numpy float32 array (H, W) into a row-major std::vector<float>.
std::vector<float> array2d_to_vector(const py::array_t<float, py::array::c_style | py::array::forcecast>& arr,
                                      std::size_t& height,
                                      std::size_t& width) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument("expected a 2D (H, W) array");
    }
    height = static_cast<std::size_t>(arr.shape(0));
    width = static_cast<std::size_t>(arr.shape(1));
    const float* data = arr.data();
    return std::vector<float>(data, data + height * width);
}

py::array_t<float> vector_to_array2d(const std::vector<float>& buf, std::size_t height, std::size_t width) {
    py::array_t<float> result({height, width});
    std::copy(buf.begin(), buf.end(), result.mutable_data());
    return result;
}

}  // namespace

// estimate_flow(frame_a: (H,W) float32, frame_b: (H,W) float32) -> (u, v)
static py::tuple py_estimate_flow(py::array_t<float, py::array::c_style | py::array::forcecast> frame_a,
                                   py::array_t<float, py::array::c_style | py::array::forcecast> frame_b) {
    std::size_t ha, wa, hb, wb;
    std::vector<float> a = array2d_to_vector(frame_a, ha, wa);
    std::vector<float> b = array2d_to_vector(frame_b, hb, wb);
    if (ha != hb || wa != wb) {
        throw std::invalid_argument("frame_a and frame_b must have the same (H, W) shape");
    }

    std::vector<float> u, v;
    nowcast::estimate_flow(a, b, ha, wa, u, v);

    return py::make_tuple(vector_to_array2d(u, ha, wa), vector_to_array2d(v, ha, wa));
}

// advect_frames(frames: (T,H,W) float32, n_steps: int) -> (n_steps, H, W) float32
static py::array_t<float> py_advect_frames(py::array_t<float, py::array::c_style | py::array::forcecast> frames,
                                            int n_steps) {
    if (frames.ndim() != 3) {
        throw std::invalid_argument("frames must be a 3D (T, H, W) array");
    }
    const std::size_t T = static_cast<std::size_t>(frames.shape(0));
    const std::size_t H = static_cast<std::size_t>(frames.shape(1));
    const std::size_t W = static_cast<std::size_t>(frames.shape(2));
    if (T < 2) {
        throw std::invalid_argument("frames must have at least 2 time steps to estimate flow");
    }
    if (n_steps <= 0) {
        throw std::invalid_argument("n_steps must be positive");
    }

    const float* data = frames.data();
    const std::size_t plane = H * W;

    std::vector<float> frame_second_last(data + (T - 2) * plane, data + (T - 1) * plane);
    std::vector<float> frame_last(data + (T - 1) * plane, data + T * plane);

    std::vector<float> u, v;
    nowcast::estimate_flow(frame_second_last, frame_last, H, W, u, v);

    std::vector<float> out;
    nowcast::advect(frame_last, u, v, H, W, n_steps, out);

    py::array_t<float> result({static_cast<std::size_t>(n_steps), H, W});
    std::copy(out.begin(), out.end(), result.mutable_data());
    return result;
}

PYBIND11_MODULE(nowcast_cpp, m) {
    m.doc() = "C++ optical-flow nowcasting baseline (Horn-Schunck flow + "
              "semi-Lagrangian advection), exposed via pybind11. See "
              "docs/technical_flow.md section 4 for the frozen contract.";

    m.def("estimate_flow", &py_estimate_flow, py::arg("frame_a"), py::arg("frame_b"),
          "Estimate dense optical flow (u, v) from frame_a to frame_b. "
          "Both frames are float32 (H, W) arrays; returns a tuple of two "
          "float32 (H, W) arrays (u = horizontal/column displacement, "
          "v = vertical/row displacement).");

    m.def("advect_frames", &py_advect_frames, py::arg("frames"), py::arg("n_steps"),
          "Estimate flow from the last two frames of `frames` (T, H, W) "
          "float32, most recent frame last, then semi-Lagrangian-advect "
          "the last frame forward n_steps steps using that constant flow "
          "field with bilinear interpolation. Returns float32 "
          "(n_steps, H, W).");
}
