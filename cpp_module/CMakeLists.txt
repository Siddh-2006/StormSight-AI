cmake_minimum_required(VERSION 3.15)
project(nowcast_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Locate pybind11's CMake config. Prefer an explicit -Dpybind11_DIR=...
# (from `python -m pybind11 --cmakedir`) passed on the command line; fall
# back to asking the active Python interpreter directly so a plain
# `cmake -B build` works too as long as the venv is active / on PATH.
if(NOT pybind11_DIR)
  if(NOT Python_EXECUTABLE AND NOT Python3_EXECUTABLE)
    find_program(PYTHON_EXE NAMES python python3)
  else()
    set(PYTHON_EXE ${Python_EXECUTABLE})
  endif()
  if(PYTHON_EXE)
    execute_process(
      COMMAND ${PYTHON_EXE} -m pybind11 --cmakedir
      OUTPUT_VARIABLE pybind11_DIR_GUESS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE PYBIND11_DIR_RESULT
    )
    if(PYBIND11_DIR_RESULT EQUAL 0 AND pybind11_DIR_GUESS)
      set(pybind11_DIR ${pybind11_DIR_GUESS})
    endif()
  endif()
endif()

find_package(pybind11 CONFIG REQUIRED)

# --- Windows + MSYS2/MinGW-w64 g++ linker workaround -----------------------
#
# On this toolchain (MSYS2 UCRT64 GCC 13.2.0 + binutils ld 2.42), linking a
# pybind11 extension module through the normal `g++ ... -shared` driver
# (which delegates to `collect2`) reproducibly fails with
# "collect2.exe: error: ld returned 116 exit status" and no further
# diagnostic, for *any* pybind11-based translation unit (verified with a
# minimal single-function PYBIND11_MODULE — plain C and plain-C++ shared
# libraries link fine through the same collect2 path). Invoking binutils
# `ld.exe` directly with the exact same arguments collect2 would have used
# succeeds without issue, so `collect2` itself (not GCC's compiler front
# end, not `ld`, not pybind11/Python.h) is the broken link in the chain on
# this setup. This appears to be a known class of MinGW `collect2` bug
# triggered by object files with very large symbol tables (pybind11's
# heavy template instantiation produces exactly that), independent of the
# exact GCC version.
#
# Workaround: on Windows with the GNU toolchain, bypass `collect2` entirely
# for the final link step by overriding CMake's shared-module link rule to
# invoke `ld.exe` directly, reconstructing the same startup
# objects/libraries GCC would normally pass. We also statically link
# libstdc++/libgcc/libwinpthread (`-Bstatic ... --start-group ... --end-group`)
# so the resulting .pyd has no MinGW runtime DLL dependencies at all beyond
# python3XX.dll and standard Windows system DLLs — this sidesteps a
# separate, unrelated issue where Python 3.8+'s restricted DLL search path
# on Windows won't find libgcc_s_seh-1.dll / libstdc++-6.dll /
# libwinpthread-1.dll next to the .pyd or even on PATH without an explicit
# os.add_dll_directory() call.
#
# If a future MSYS2/binutils update fixes the underlying collect2 bug, this
# whole block can be deleted and pybind11_add_module's default link line
# will work unmodified (it already did for non-pybind11 C/C++ shared libs
# on this same machine).
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-prog-name=ld
    OUTPUT_VARIABLE MINGW_LD OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=dllcrt2.o
    OUTPUT_VARIABLE MINGW_DLLCRT2 OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=crtbegin.o
    OUTPUT_VARIABLE MINGW_CRTBEGIN OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=crtend.o
    OUTPUT_VARIABLE MINGW_CRTEND OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-libgcc-file-name
    OUTPUT_VARIABLE MINGW_LIBGCC OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpmachine
    OUTPUT_VARIABLE MINGW_TARGET_TRIPLE OUTPUT_STRIP_TRAILING_WHITESPACE)

  get_filename_component(MINGW_GCC_LIB_DIR ${MINGW_LIBGCC} DIRECTORY)
  get_filename_component(MINGW_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
  get_filename_component(MINGW_ROOT_DIR ${MINGW_BIN_DIR} DIRECTORY)
  set(MINGW_LIB_DIR "${MINGW_ROOT_DIR}/lib")
  set(MINGW_TARGET_LIB_DIR "${MINGW_ROOT_DIR}/${MINGW_TARGET_TRIPLE}/lib")

  if(EXISTS ${MINGW_LD} AND EXISTS ${MINGW_DLLCRT2})
    message(STATUS "nowcast_cpp: using direct-ld link workaround (${MINGW_LD}) "
                    "to avoid a collect2 crash linking pybind11 modules on this toolchain")
    set(CMAKE_CXX_CREATE_SHARED_MODULE
      "${MINGW_LD} -m i386pep --shared -Bdynamic -e DllMainCRTStartup --enable-auto-image-base -o <TARGET> \
${MINGW_DLLCRT2} ${MINGW_CRTBEGIN} \
-L${MINGW_GCC_LIB_DIR} -L${MINGW_LIB_DIR} -L${MINGW_TARGET_LIB_DIR} \
<OBJECTS> <LINK_LIBRARIES> \
-lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 \
-Bstatic -lstdc++ --start-group -lmingw32 -lgcc -lgcc_eh -lwinpthread -lmingwex -lmoldname -lmsvcrt --end-group \
-Bdynamic -lkernel32 -ladvapi32 -lshell32 -luser32 -lkernel32 \
${MINGW_CRTEND}"
    )
  else()
    message(WARNING "nowcast_cpp: could not locate MinGW ld/CRT objects for the "
                     "direct-ld link workaround; falling back to the default "
                     "g++ link line, which is known to crash with collect2 on "
                     "this toolchain for pybind11 modules. See CMakeLists.txt.")
  endif()
endif()
# --- end Windows/MinGW workaround ------------------------------------------

pybind11_add_module(nowcast_cpp
  src/bindings.cpp
  src/advect.cpp
)

target_include_directories(nowcast_cpp PRIVATE include)

# Reasonable optimization for the numeric core in Release builds.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(nowcast_cpp PRIVATE -O3)
endif()

# Place the built extension at cpp_module/ (repo-relative), so it's easy to
# import directly (e.g. via sys.path.insert) without an install step, in
# addition to whatever pybind11_add_module's default output dir already is.
set_target_properties(nowcast_cpp PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Needed for the scikit-build-core / `pip install .` path (see
# pyproject.toml): it packages whatever this install() rule puts at the
# wheel root, alongside wheel.install-dir = "." in pyproject.toml. Harmless
# for the plain `cmake --build` workflow, which never runs `cmake --install`.
install(TARGETS nowcast_cpp DESTINATION .)
