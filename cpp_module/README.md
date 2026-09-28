# cpp_module — `nowcast_cpp`

C++ optical-flow nowcasting baseline (Horn-Schunck-style dense flow +
semi-Lagrangian advection), exposed to Python as `nowcast_cpp` via pybind11.
This implements the **frozen contract** in `docs/technical_flow.md` §4
exactly:

```python
def estimate_flow(frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """frame_a, frame_b: float32 (H, W). Returns (u, v), each float32 (H, W):
    u = horizontal/column displacement, v = vertical/row displacement,
    describing motion from frame_a to frame_b."""

def advect_frames(frames: np.ndarray, n_steps: int) -> np.ndarray:
    """frames: float32 (T, H, W), most recent frame last.
    Returns float32 (n_steps, H, W): estimates flow from the last two input
    frames, then semi-Lagrangian-advects the last frame forward n_steps
    times using that constant flow field, with bilinear interpolation."""
```

Both functions are implemented from scratch in plain C++17
(`src/advect.cpp` / `include/advect.hpp`) using only `std::vector<float>`
buffers — no OpenCV, no Eigen, no other numeric dependency. `src/bindings.cpp`
is the pybind11 glue.

## Verified environment

- Windows 11, MSYS2 **UCRT64** `g++` 13.2.0 / binutils `ld` 2.42 (`C:\msys64\ucrt64\bin`)
- CMake 4.4.3
- Python 3.13.7 (the standard python.org / MSVC-ABI build at `C:\Python313`,
  same interpreter the project's `uv venv .venv` uses)
- venv at repo root `.venv/`, created with `uv venv`, with
  `uv pip install pybind11 numpy` already run in it (this doc assumes that)

Everything below was actually run and passed in this exact environment —
this is not a "should work" writeup. See **Known issue on this toolchain**
below for the one non-obvious thing you need.

## Known issue on this toolchain: `collect2` crash linking pybind11 modules

On this MSYS2 UCRT64 GCC 13.2.0 + binutils 2.42 setup, linking a pybind11
extension module through the normal compiler-driver route
(`g++ ... -shared -o module.pyd ...`, which internally delegates to
`collect2.exe`) **reproducibly fails**:

```
collect2.exe: error: ld returned 116 exit status
```

with no further diagnostic. This was isolated (not guessed) by testing, in
order:

1. Linking a plain C Python C-API extension (no pybind11, no C++) against
   `python313.lib` through `gcc` → **works**.
2. Linking a trivial C++ shared library (no pybind11) through `g++` →
   **works**.
3. Linking the smallest possible pybind11 module
   (`PYBIND11_MODULE(m, mod) { mod.def("add", ...); }`) through `g++` →
   **fails**, deterministically, every time, with exactly the error above.
4. Taking the *exact* link command line `g++ -v` shows it's handing to
   `collect2`, and running the equivalent `ld.exe` invocation **directly**
   (bypassing `collect2`) → **works**, produces a valid, correctly-loadable
   `.pyd`.

So the break is specifically in `collect2`'s handling of pybind11's
heavily-templated object files on this GCC/binutils build — not in `g++`
compiling, not in `ld` itself, not in pybind11/Python.h, and not in
`python313.lib` (the MSVC-built Python's import library links against
MinGW-produced objects without issue otherwise).

**Fix applied in `CMakeLists.txt`**: on `WIN32` with the GNU compiler, the
build overrides CMake's `CMAKE_CXX_CREATE_SHARED_MODULE` rule to invoke
`ld.exe` directly for the final link step, reconstructing the same CRT
startup objects (`dllcrt2.o`, `crtbegin.o`/`crtend.o`) and system libraries
GCC would normally pass, all discovered dynamically via
`${CMAKE_CXX_COMPILER} -print-file-name=...` (no hardcoded MSYS2 version
numbers). It also **statically links** `libstdc++`/`libgcc`/`libwinpthread`
(`-Bstatic ... --start-group ... --end-group`), which sidesteps a second,
unrelated issue: Python 3.8+'s restricted DLL search path on Windows won't
find `libgcc_s_seh-1.dll` / `libstdc++-6.dll` / `libwinpthread-1.dll` next to
the `.pyd` (or even on `PATH`) without an explicit
`os.add_dll_directory()` call at import time. Static-linking the runtime
means the built `.pyd` depends on nothing but `python3XX.dll` and standard
Windows system DLLs (verified with `objdump -p`), so a plain
`import nowcast_cpp` works with zero extra setup.

If a future MSYS2/binutils update fixes the underlying `collect2` bug, this
whole block in `CMakeLists.txt` (clearly marked) can simply be deleted; the
default `pybind11_add_module` link line will work unmodified (it already
does for non-pybind11 C/C++ shared libraries on this same machine).

## Build (verified): plain CMake + MinGW Makefiles

This is the path actually built and tested end-to-end.

```bash
# From repo root, with the venv already created (uv venv .venv) and
# pybind11 + numpy already installed in it (uv pip install pybind11 numpy):

cd cpp_module
cmake -S . -B build -G "MinGW Makefiles" \
  -DPYTHON_EXECUTABLE="../.venv/Scripts/python.exe" \
  -Dpybind11_DIR="$(../.venv/Scripts/python.exe -m pybind11 --cmakedir)"

cmake --build build -j4
```

This produces `cpp_module/nowcast_cpp.cp313-win_amd64.pyd` (the CMakeLists
sets the output directory to `cpp_module/` itself, i.e. repo-relative, so
it's importable without an install step, e.g. from `backend/` via
`sys.path.insert(0, "<repo>/cpp_module")`, or just copy/symlink the `.pyd`
next to wherever it needs to be imported from).

Verify it:

```bash
../.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, '.')
import nowcast_cpp
print(nowcast_cpp.__file__)
"
```

**Note on `-DPYTHON_EXECUTABLE`**: pybind11's CMake config uses the legacy
`find_package(PythonInterp)`/`find_package(PythonLibs)` path (not modern
`find_package(Python)`), so the variable to force it onto the venv's
interpreter is the legacy `PYTHON_EXECUTABLE`, not `Python_EXECUTABLE`. If
you omit it, CMake may pick up a *different* Python on `PATH` (e.g. MSYS2's
own `/ucrt64/bin/python3.exe` if that's installed) instead of the venv's —
harmless for the build itself, but the resulting `.pyd` will be built for
the wrong Python version/ABI and fail to import from the venv.

## Alternative (also verified): `pip install .` via scikit-build-core

A `pyproject.toml` is included so the extension can also be built as a
normal pip-installable package, using the *same* `CMakeLists.txt` (same
collect2 workaround). This was also built and verified in this environment:

```bash
# from repo root, venv active / using its python directly:
uv pip install --python .venv/Scripts/python.exe scikit-build-core pybind11 numpy

# CMAKE_GENERATOR is needed because this machine has neither Ninja nor MSVC
# installed, and scikit-build-core doesn't reliably auto-select MinGW
# Makefiles on its own:
CMAKE_GENERATOR="MinGW Makefiles" uv pip install --python .venv/Scripts/python.exe \
  --no-build-isolation ./cpp_module
```

(`--no-build-isolation` is used here because pybind11/scikit-build-core are
already installed in the venv and this avoids uv/pip spinning up a second,
network-fetching isolated build env — remove it if you'd rather let it
build fully isolated, which should also work but was not the path
re-verified here.)

This installs `nowcast_cpp.cp313-win_amd64.pyd` straight into the venv's
`site-packages`, so `import nowcast_cpp` works from anywhere using that
venv, with no `sys.path` manipulation:

```bash
.venv/Scripts/python.exe -c "import nowcast_cpp; print(nowcast_cpp.__file__)"
```

Both build paths were verified to produce a working, importable module and
both pass `test_advect.py`. The plain CMake path is documented first/in more
depth since it's the one the rest of this troubleshooting doc was written
against; the scikit-build-core path is a thin wrapper around the same
CMakeLists.txt and inherits the same fix.

## Caveats / things to know

- **This is a from-scratch, hackathon-grade Horn-Schunck implementation**,
  not a production optical-flow library. It's tuned (see
  `include/advect.hpp` defaults: `alpha=0.01`, `iterations=200`) for inputs
  normalized to `[0, 1]` (as the project's grid convention requires — see
  `docs/technical_flow.md` §2). If someone changes the normalization range
  fed into `estimate_flow`, `alpha` will likely need retuning (Horn-Schunck's
  smoothness weight has to be scaled to the gradient magnitude of the input;
  the very first tuning attempt at `alpha=50` — a reasonable value for
  `[0, 255]`-scale imagery from classic Horn-Schunck literature — produced
  near-zero flow on our `[0, 1]`-scale data, which is exactly what
  `test_advect.py` caught).
- The CMake link workaround hardcodes nothing MSYS2-version-specific; it
  re-derives every path from `${CMAKE_CXX_COMPILER}` at configure time, so
  it should keep working across MSYS2 package updates as long as the
  underlying `collect2` bug persists, and will simply not trigger (the
  `if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")` guard) on
  Linux/macOS or an MSVC build, where the normal `pybind11_add_module` link
  line is used unmodified.
- Not tested against MSVC (`cl.exe`)/Visual Studio generators, or against
  clang, since none were available in this environment — only MSYS2 UCRT64
  g++ was verified, per the task's stated environment.

## Testing

```bash
../.venv/Scripts/python.exe test_advect.py
# or, with pytest installed:
../.venv/Scripts/python.exe -m pytest test_advect.py -v -s
```

`test_advect.py` generates a synthetic Gaussian-blob sequence translating at
a known constant pixel velocity (pure NumPy, no dependency on the rest of
the repo), calls `advect_frames`, computes the intensity-weighted centroid
of each predicted frame, and asserts it's within 3 pixels of the
analytically-expected position (`last_observed_position + step * velocity`).
It also sanity-checks `estimate_flow`'s output shape/dtype and that the
estimated flow near the blob points in the right direction with roughly the
right magnitude.

**Confirmed passing in this environment** (both direct run and via
pytest), e.g.:

```
step 1: expected center=(30.00, 26.00) got=(29.93, 25.91) err=(0.07, 0.09)
step 2: expected center=(31.50, 28.00) got=(31.26, 27.69) err=(0.24, 0.31)
step 3: expected center=(33.00, 30.00) got=(32.48, 29.30) err=(0.52, 0.70)
step 4: expected center=(34.50, 32.00) got=(33.56, 30.76) err=(0.94, 1.24)
PASS: max row error 0.936px, max col error 1.239px (tolerance 3.0px)
```

## Files

```
cpp_module/
├── CMakeLists.txt        # build config incl. the collect2 workaround above
├── pyproject.toml        # scikit-build-core config for `pip install .`
├── include/advect.hpp    # estimate_flow / bilinear_sample / advect declarations
├── src/advect.cpp        # from-scratch Horn-Schunck flow + semi-Lagrangian advection
├── src/bindings.cpp      # pybind11 glue exposing the `nowcast_cpp` module
├── test_advect.py        # correctness test (synthetic moving Gaussian blob)
└── README.md             # this file
```
