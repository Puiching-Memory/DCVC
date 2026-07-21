# DCVC CPU ONNX codec: a standalone, cross-platform (Linux/Windows), CPU-only
# DCVC-RT I-frame codec built on ONNX Runtime + the native C rANS library.
cmake_minimum_required(VERSION 3.18)
project(dcvc_onnx C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Directory holding the runtime ONNX models + CDF tables (onnx/models by default).
set(DCVC_ONNX_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models"
    CACHE PATH "Directory with *.onnx models and *.npy CDF tables")

# ---- Cross-platform compile flags ----
if(MSVC)
    add_compile_options(/W3 /utf-8 /D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra)
    # Look up the ONNX Runtime shared lib sitting next to the executable.
    set(CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

include(FetchContent)

# ===========================================================================
# Fetch a prebuilt ONNX Runtime for the current platform (Linux/Windows x64).
# ===========================================================================
set(ORT_VERSION "1.19.0" CACHE STRING "ONNX Runtime version to fetch")
if(WIN32)
    set(ORT_PACKAGE_NAME "onnxruntime-win-x64-${ORT_VERSION}")
    set(_ORT_EXT "zip")
else()
    set(ORT_PACKAGE_NAME "onnxruntime-linux-x64-${ORT_VERSION}")
    set(_ORT_EXT "tgz")
endif()

# Where the ONNX Runtime archive is obtained from. Resolution order:
#   1. DCVC_ORT_ARCHIVE  -> use a local archive file you already downloaded
#                          (fastest; lets you fetch via any mirror/proxy first).
#   2. DCVC_ORT_URL      -> download from this exact URL (a mirror, e.g.
#                          https://gh-proxy.com/https://github.com/...).
#   3. DCVC_ORT_URL_BASE -> prefix prepended to the default github.com path
#                          (a mirror host, e.g. https://gh-proxy.com/).
#   4. default           -> GitHub releases directly.
set(_ORT_DEFAULT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_PACKAGE_NAME}.${_ORT_EXT}")
if(DCVC_ORT_URL_BASE AND NOT DCVC_ORT_URL)
    string(REGEX REPLACE "^https?://" "" _ORT_TAIL "${_ORT_DEFAULT_URL}")
    set(ORT_URL "${DCVC_ORT_URL_BASE}${_ORT_TAIL}")
else()
    set(ORT_URL "${_ORT_DEFAULT_URL}" CACHE STRING "ONNX Runtime download URL")
endif()

set(ORT_ROOT "${CMAKE_BINARY_DIR}/${ORT_PACKAGE_NAME}")
set(ORT_STAMP "${CMAKE_BINARY_DIR}/${ORT_PACKAGE_NAME}.stamp")

if(NOT EXISTS "${ORT_STAMP}")
    set(ORT_ARCHIVE_DST "${CMAKE_BINARY_DIR}/${ORT_PACKAGE_NAME}.${_ORT_EXT}")
    if(DCVC_ORT_ARCHIVE AND EXISTS "${DCVC_ORT_ARCHIVE}")
        message(STATUS "ONNX Runtime: using local archive ${DCVC_ORT_ARCHIVE}")
        set(ORT_ARCHIVE_DST "${DCVC_ORT_ARCHIVE}")
    else()
        message(STATUS "ONNX Runtime: downloading from ${ORT_URL}")
        file(DOWNLOAD "${ORT_URL}" "${ORT_ARCHIVE_DST}" SHOW_PROGRESS STATUS _ort_dl)
        list(GET _ort_dl 0 _ort_status)
        if(NOT _ort_status EQUAL 0)
            file(REMOVE "${ORT_ARCHIVE_DST}")
            message(FATAL_ERROR "Download failed (status=${_ort_dl}). Set -DDCVC_ORT_ARCHIVE=<local.zip> or -DDCVC_ORT_URL_BASE=https://gh-proxy.com/ to use a mirror.")
        endif()
    endif()
    file(ARCHIVE_EXTRACT INPUT "${ORT_ARCHIVE_DST}" DESTINATION "${CMAKE_BINARY_DIR}")
    file(WRITE "${ORT_STAMP}" "${ORT_URL}\n")
endif()

set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB_DIR "${ORT_ROOT}/lib")
if(WIN32)
    file(GLOB ORT_RUNTIME_LIBS "${ORT_LIB_DIR}/onnxruntime*.dll")
    if(MINGW)
        # MinGW's GNU ld links directly against the DLL and auto-generates the
        # import data; the MSVC .lib is not used in this cross-compile path.
        set(ORT_IMPLIB "${ORT_LIB_DIR}/onnxruntime.dll")
    else()
        set(ORT_IMPLIB "${ORT_LIB_DIR}/onnxruntime.lib")
    endif()
else()
    set(ORT_IMPLIB "${ORT_LIB_DIR}/libonnxruntime.so")
    file(GLOB ORT_RUNTIME_LIBS "${ORT_LIB_DIR}/libonnxruntime.so*")
endif()

find_package(Threads REQUIRED)

# ===========================================================================
# rANS entropy-coding library (pure C++/C, self-contained under onnx/rans).
# ===========================================================================
set(DCVC_RANS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rans")
add_library(dcvc_rans_cpu STATIC
    ${DCVC_RANS_DIR}/rans.cpp
    ${DCVC_RANS_DIR}/rans_c.cpp
    ${DCVC_RANS_DIR}/pmf_cdf.cpp
)
target_include_directories(dcvc_rans_cpu PUBLIC
    ${DCVC_RANS_DIR}
)
target_link_libraries(dcvc_rans_cpu PUBLIC Threads::Threads)
if(NOT WIN32)
    target_link_libraries(dcvc_rans_cpu PUBLIC ${CMAKE_DL_LIBS})
endif()

# ===========================================================================
# Helper: add an ONNX-backed executable with all the boilerplate wired up.
# ===========================================================================
function(dcvc_onnx_add_executable name)
    set(multi_args SRCS)
    cmake_parse_arguments(ARG "" "" "${multi_args}" ${ARGN})

    add_executable(${name} ${ARG_SRCS})

    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${ORT_INCLUDE_DIR}
    )
    target_link_directories(${name} PRIVATE ${ORT_LIB_DIR})
    target_link_libraries(${name} PRIVATE dcvc_rans_cpu ${ORT_IMPLIB})
    if(NOT WIN32)
        target_link_libraries(${name} PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    endif()

    if(MINGW)
        # Statically link the MinGW C/C++/pthread runtimes so the produced
        # .exe files only depend on onnxruntime.dll + Windows system DLLs
        # (no libgcc_s_seh-1 / libstdc++-6 / libwinpthread-1 to ship).
        target_link_options(${name} PRIVATE -static)
    endif()

    # Place the ONNX Runtime shared lib(s) next to the executable.
    foreach(lib ${ORT_RUNTIME_LIBS})
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${lib}" "$<TARGET_FILE_DIR:${name}>")
    endforeach()
endfunction()

# ===========================================================================
# Codec sources
# ===========================================================================
set(DCVC_NPY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/npy_reader.c)
# Shared helpers used by every test executable (model-dir auto-detection).
set(DCVC_COMMON_SRC ${DCVC_NPY_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/src/model_dir.c)

set(DCVC_CODEC_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_engine.cpp
    ${DCVC_COMMON_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_ar_codec.c
)
set(DCVC_PIPELINE_SRCS
    ${DCVC_CODEC_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_intra_pipeline.c
)

set(DCVC_INTER_PIPELINE_SRCS
    ${DCVC_CODEC_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_intra_pipeline.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_inter_pipeline.c
)

dcvc_onnx_add_executable(test_intra_analysis
    SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_engine.cpp
         ${DCVC_COMMON_SRC}
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_intra_analysis.cpp)

dcvc_onnx_add_executable(test_ar_cpu
    SRCS ${DCVC_CODEC_SRCS}
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ar_cpu.cpp)

dcvc_onnx_add_executable(test_cpu_end2end
    SRCS ${DCVC_PIPELINE_SRCS}
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu_end2end.cpp)

dcvc_onnx_add_executable(test_cpu_inter
    SRCS ${DCVC_INTER_PIPELINE_SRCS}
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu_inter.cpp)

dcvc_onnx_add_executable(test_inter_one
    SRCS ${DCVC_INTER_PIPELINE_SRCS}
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_inter_one.cpp)

# ===========================================================================
# `dcvc_package` target: assemble a ready-to-run folder (binaries + ORT lib
# + models + CDF tables) under dist/dcvc_onnx_codec/. Works on Linux+Windows.
# ===========================================================================
set(DCVC_DIST_ROOT "${CMAKE_SOURCE_DIR}/dist")
set(DCVC_DIST_DIR "${DCVC_DIST_ROOT}/dcvc_onnx_codec")

# Short run instructions dropped into the package root.
file(WRITE "${CMAKE_BINARY_DIR}/README_PACKAGE.txt"
"DCVC CPU ONNX codec - runnable package
======================================
Contents:
  *.exe / test_*   codec test executables
  onnxruntime.dll / libonnxruntime.so*   ONNX Runtime shared library
  *.onnx / *.npy    neural-network models and entropy CDF tables

The executables auto-detect the model directory, so run them with no
arguments from this folder:
  Windows:  test_cpu_end2end.exe      (I-frame)
            test_cpu_inter.exe      (I+P frame sequence)
            test_ar_cpu.exe         (AR codec unit test)
  Linux:    ./test_cpu_end2end      (I-frame)
            ./test_cpu_inter        (I+P frame sequence)
            ./test_ar_cpu           (AR codec unit test)

Optional arguments override the defaults:
  test_cpu_end2end[.exe] [model_dir] [H] [W] [qp]   (any H,W; default 256 256 32)
  test_ar_cpu[.exe]      [model_dir] [H] [W]        (default 8 8)
")

add_custom_target(dcvc_package
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${DCVC_DIST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DCVC_DIST_DIR}"
    # Executables (generator expressions resolve the per-config output path).
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:test_intra_analysis>" "${DCVC_DIST_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:test_ar_cpu>" "${DCVC_DIST_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:test_cpu_end2end>" "${DCVC_DIST_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:test_cpu_inter>" "${DCVC_DIST_DIR}/"
    # ONNX Runtime shared library (next to the binaries).
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ORT_RUNTIME_LIBS} "${DCVC_DIST_DIR}/"
    # Models + CDF tables.
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${DCVC_ONNX_MODEL_DIR}" "${DCVC_DIST_DIR}"
    # Run instructions.
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_BINARY_DIR}/README_PACKAGE.txt" "${DCVC_DIST_DIR}/"
    DEPENDS test_intra_analysis test_ar_cpu test_cpu_end2end test_cpu_inter
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Assembling runnable package at ${DCVC_DIST_DIR}"
    VERBATIM
)
