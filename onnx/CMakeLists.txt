# DCVC CPU ONNX codec: a standalone, cross-platform (Linux/Windows) DCVC-UF
# I-frame + P-chunk codec built on ONNX Runtime + the native C rANS library.
# CPU by default; configure with -DDCVC_ORT_GPU=ON to fetch the CUDA-enabled
# ONNX Runtime package and enable the CUDA/TensorRT execution providers.
cmake_minimum_required(VERSION 3.18)
project(dcvc_onnx C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Keep every generated artifact below one predictable root. Toolchain-specific
# tags prevent native MSVC and MinGW packages from overwriting each other.
set(DCVC_OUTPUT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/out" CACHE PATH
    "Root for generated build-independent artifacts")
if(MINGW)
    set(_DCVC_DEFAULT_ARTIFACT_TAG "windows-x64-mingw")
elseif(MSVC)
    set(_DCVC_DEFAULT_ARTIFACT_TAG "windows-x64-msvc")
else()
    set(_DCVC_DEFAULT_ARTIFACT_TAG "linux-x64")
endif()
set(DCVC_ARTIFACT_TAG "${_DCVC_DEFAULT_ARTIFACT_TAG}" CACHE STRING
    "Platform/toolchain tag used below DCVC_OUTPUT_ROOT/runnable")
if(NOT DCVC_ARTIFACT_TAG MATCHES "^[A-Za-z0-9][A-Za-z0-9._-]*$")
    message(FATAL_ERROR "Invalid DCVC_ARTIFACT_TAG '${DCVC_ARTIFACT_TAG}'")
endif()
get_filename_component(DCVC_OUTPUT_ROOT "${DCVC_OUTPUT_ROOT}" ABSOLUTE
    BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(DCVC_RUNNABLE_DIR
    "${DCVC_OUTPUT_ROOT}/runnable/${DCVC_ARTIFACT_TAG}")
message(STATUS "DCVC output root: ${DCVC_OUTPUT_ROOT}")

# Runtime model packs included by dcvc_package. The value is a semicolon-
# separated name=directory list. When empty, resolution directories matching
# models_<digits>p (for example models_720p and models_2160p) are discovered.
set(DCVC_ONNX_MODEL_PACKS "" CACHE STRING
    "Model packs as name=directory entries; empty means discover models_<resolution>")

set(_DCVC_MODEL_PACK_SPECS ${DCVC_ONNX_MODEL_PACKS})
if(NOT _DCVC_MODEL_PACK_SPECS)
    file(GLOB _DCVC_MODEL_PACK_CANDIDATES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        LIST_DIRECTORIES true "${CMAKE_CURRENT_SOURCE_DIR}/models_*p")
    foreach(_candidate IN LISTS _DCVC_MODEL_PACK_CANDIDATES)
        if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_candidate}"
           AND _candidate MATCHES "^models_([0-9]+p)$")
            list(APPEND _DCVC_MODEL_PACK_SPECS
                "${CMAKE_MATCH_1}=${CMAKE_CURRENT_SOURCE_DIR}/${_candidate}")
        endif()
    endforeach()
    list(SORT _DCVC_MODEL_PACK_SPECS COMPARE NATURAL)
endif()
set(_DCVC_MODEL_PACK_NAMES)
set(_DCVC_MODEL_PACKAGE_COMMANDS)
set(_DCVC_MODEL_CONTENTS "")
set(_DCVC_MODEL_QUICK_TESTS "")
if(NOT _DCVC_MODEL_PACK_SPECS)
    message(WARNING
        "No model packs found; library targets remain available, but "
        "dcvc_package will fail. Create models_<resolution> or set "
        "DCVC_ONNX_MODEL_PACKS=name=/path/to/models")
    list(APPEND _DCVC_MODEL_PACKAGE_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E echo
            "error: no DCVC model packs configured"
        COMMAND ${CMAKE_COMMAND} -E false)
endif()
foreach(_spec IN LISTS _DCVC_MODEL_PACK_SPECS)
    if(NOT _spec MATCHES "^([A-Za-z0-9][A-Za-z0-9._-]*)=(.+)$")
        message(FATAL_ERROR "Invalid model pack '${_spec}'; expected name=directory")
    endif()
    set(_pack_name "${CMAKE_MATCH_1}")
    set(_pack_dir "${CMAKE_MATCH_2}")
    if(NOT IS_ABSOLUTE "${_pack_dir}")
        get_filename_component(_pack_dir "${_pack_dir}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    if(_pack_name IN_LIST _DCVC_MODEL_PACK_NAMES)
        message(FATAL_ERROR "Duplicate model pack name '${_pack_name}'")
    endif()
    if(NOT EXISTS "${_pack_dir}/intra_analysis_standard.onnx")
        message(FATAL_ERROR
            "Model pack '${_pack_name}' is missing "
            "${_pack_dir}/intra_analysis_standard.onnx")
    endif()
    list(APPEND _DCVC_MODEL_PACK_NAMES "${_pack_name}")
    string(APPEND _DCVC_MODEL_CONTENTS
        "  models/${_pack_name}/       neural-network models + CDF tables\n")
    list(APPEND _DCVC_MODEL_PACKAGE_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${DCVC_RUNNABLE_DIR}/models/${_pack_name}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${_pack_dir}"
            "${DCVC_RUNNABLE_DIR}/models/${_pack_name}")
    if(_pack_name STREQUAL "720p")
        string(APPEND _DCVC_MODEL_QUICK_TESTS
            "  Linux:    ./test_cpu_end2end models/720p 768 1280 32\n"
            "  Windows:  test_cpu_end2end.exe models\\720p 768 1280 32\n")
    elseif(_pack_name STREQUAL "1080p")
        string(APPEND _DCVC_MODEL_QUICK_TESTS
            "  Linux:    ./test_cpu_end2end models/1080p 1088 1920 32\n"
            "  Windows:  test_cpu_end2end.exe models\\1080p 1088 1920 32\n")
    endif()
endforeach()
message(STATUS "DCVC model packs: ${_DCVC_MODEL_PACK_NAMES}")

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
set(ORT_VERSION "1.27.0" CACHE STRING "ONNX Runtime version to fetch")
# ON: fetch the GPU package (CUDA + TensorRT providers; requires matching
# CUDA/cuDNN, and TensorRT for the TensorRT EP, installed on the system).
# OFF: fetch the CPU-only package.
option(DCVC_ORT_GPU "Fetch the CUDA-enabled ONNX Runtime package" OFF)
# Build FXP CUDA kernels + register CUDAExecutionProvider custom ops.
# Independent of DCVC_ORT_GPU: kernels can be tested without ORT GPU EP.
option(DCVC_FXP_CUDA "Build CUDA backends for com.dcvc FXP custom ops" ON)
# When DCVC_ORT_GPU=ON, select the CUDA major version of the ORT package
# (ORT 1.27+ splits GPU packages by CUDA version: gpu_cuda12 / gpu_cuda13).
set(DCVC_ORT_CUDA_VERSION "12" CACHE STRING "CUDA major version for the GPU-enabled ORT package (12 or 13)")
if(WIN32)
    set(_ORT_PLATFORM "win-x64")
    set(_ORT_EXT "zip")
else()
    set(_ORT_PLATFORM "linux-x64")
    set(_ORT_EXT "tgz")
endif()
if(DCVC_ORT_GPU)
    set(_ORT_PLATFORM "${_ORT_PLATFORM}-gpu_cuda${DCVC_ORT_CUDA_VERSION}")
endif()
set(ORT_PACKAGE_NAME "onnxruntime-${_ORT_PLATFORM}-${ORT_VERSION}")

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
    # GPU packages also ship libonnxruntime_providers_{shared,cuda,tensorrt}.so,
    # which ORT dlopens at EP append time; they must sit next to the binary.
    file(GLOB ORT_RUNTIME_LIBS "${ORT_LIB_DIR}/libonnxruntime*.so*")
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
set_target_properties(dcvc_rans_cpu PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
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
    if(DCVC_FXP_CUDA_ENABLED)
        target_compile_definitions(${name} PRIVATE DCVC_FXP_CUDA=1)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(${name} PRIVATE CUDA::cudart)
        set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
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

set(DCVC_FXP_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_conv.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/det_conv.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_wsrelu.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_scale_index.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/ort_custom_ops.cpp
)

set(DCVC_FXP_CUDA_ENABLED FALSE)
if(DCVC_FXP_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        list(APPEND DCVC_FXP_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_cuda.cu)
        set(DCVC_FXP_CUDA_ENABLED TRUE)
        message(STATUS "DCVC FXP CUDA: enabled (${CMAKE_CUDA_COMPILER})")
    else()
        message(WARNING "DCVC_FXP_CUDA=ON but no CUDA compiler found; building CPU-only FXP")
    endif()
endif()

# Layout-friendly FXP C kernels: enable auto-vectorization (no intrinsics).
if(NOT MSVC)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_conv.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_wsrelu.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_scale_index.c
        PROPERTIES COMPILE_OPTIONS "-O3;-mavx2"
    )
    # det_conv.c: float64 accumulation — disable FMA contraction for
    # cross-platform bit-exactness.
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/det_conv.c
        PROPERTIES COMPILE_OPTIONS "-O3;-mavx2;-ffp-contract=off;-fno-fast-math"
    )
endif()

if(DCVC_FXP_CUDA_ENABLED)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/ort_custom_ops.cpp
        PROPERTIES COMPILE_DEFINITIONS "DCVC_FXP_CUDA=1"
    )
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_cuda.cu
        PROPERTIES COMPILE_OPTIONS "--expt-relaxed-constexpr;-fmad=false"
    )
endif()

set(DCVC_CODEC_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_engine.cpp
    ${DCVC_FXP_SRCS}
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

# ===========================================================================
# dcvc — commercial SDK shared library (public API in include/dcvc/).
# Bundles the FXP fixed-point ops, rANS entropy codec, ONNX Runtime engine
# and the intra/inter pipelines behind a stable C facade (sdk/).
# ===========================================================================
option(DCVC_BUILD_SDK "Build the libdcvc shared library + SDK package" ON)

if(DCVC_BUILD_SDK)
    set(DCVC_VERSION_MAJOR 1)
    set(DCVC_VERSION_MINOR 0)
    set(DCVC_VERSION_PATCH 0)
    set(DCVC_SDK_SRCS
        ${DCVC_INTER_PIPELINE_SRCS}
        ${CMAKE_CURRENT_SOURCE_DIR}/sdk/dcvc_facade.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sdk/dcvc_container.c
    )
    add_library(dcvc SHARED ${DCVC_SDK_SRCS})
    target_compile_definitions(dcvc PRIVATE DCVC_EXPORTS=1)

    target_include_directories(dcvc
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/rans
            ${ORT_INCLUDE_DIR}
    )

    # CUDA backends for the FXP custom ops (optional, same gate as the tests).
    if(DCVC_FXP_CUDA_ENABLED)
        target_sources(dcvc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_cuda.cu)
        target_compile_definitions(dcvc PRIVATE DCVC_FXP_CUDA=1)
        target_link_libraries(dcvc PRIVATE CUDA::cudart)
    endif()

    target_link_libraries(dcvc PRIVATE dcvc_rans_cpu ${ORT_IMPLIB})
    target_link_directories(dcvc PRIVATE ${ORT_LIB_DIR})
    # MinGW: statically link the C/C++/pthread runtimes into the DLL so the
    # resulting dcvc.dll has no libstdc++-6 / libgcc_s / libwinpthread deps
    # (a Windows package then only needs onnxruntime.dll + the models).
    if(WIN32 AND CMAKE_C_COMPILER_ID MATCHES "GNU")
        target_link_options(dcvc PRIVATE -static)
    endif()
    if(Threads_FOUND)
        target_link_libraries(dcvc PRIVATE Threads::Threads)
    endif()
    if(NOT WIN32)
        target_link_libraries(dcvc PRIVATE ${CMAKE_DL_LIBS})
    endif()

    set_target_properties(dcvc PROPERTIES
        VERSION ${DCVC_VERSION_MAJOR}.${DCVC_VERSION_MINOR}.${DCVC_VERSION_PATCH}
        SOVERSION ${DCVC_VERSION_MAJOR}
        OUTPUT_NAME dcvc
        CXX_VISIBILITY_PRESET hidden
        C_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )

    # ---- install + exported CMake target + package ----------------------
    include(GNUInstallDirs)
    install(TARGETS dcvc
        EXPORT dcvcTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION include
    )
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/dcvc
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
    )

    install(EXPORT dcvcTargets
        FILE dcvcTargets.cmake
        NAMESPACE dcvc::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dcvc
    )
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/dcvcConfigVersion.cmake"
        VERSION ${DCVC_VERSION_MAJOR}.${DCVC_VERSION_MINOR}.${DCVC_VERSION_PATCH}
        COMPATIBILITY SameMajorVersion
    )
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dcvcConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/dcvcConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dcvc
    )
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/dcvcConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/dcvcConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dcvc
    )

    # ---- example app, links the SDK shared library ----------------------
    add_executable(dcvc_demo ${CMAKE_CURRENT_SOURCE_DIR}/examples/dcvc_demo.c)
    target_link_libraries(dcvc_demo PRIVATE dcvc)
    if(NOT WIN32)
        target_link_libraries(dcvc_demo PRIVATE m)
    endif()
    set_target_properties(dcvc_demo PROPERTIES
        BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
endif()

dcvc_onnx_add_executable(test_intra_analysis
    SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_engine.cpp
         ${DCVC_FXP_SRCS}
         ${DCVC_COMMON_SRC}
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_intra_analysis.cpp)

dcvc_onnx_add_executable(test_fxp_conv1x1
    SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_engine.cpp
         ${DCVC_FXP_SRCS}
         ${DCVC_COMMON_SRC}
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fxp_conv1x1.cpp)

dcvc_onnx_add_executable(test_fxp_scale_index
    SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_scale_index.c
         ${CMAKE_CURRENT_SOURCE_DIR}/src/npy_reader.c
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fxp_scale_index.cpp)

dcvc_onnx_add_executable(test_fxp_opt_bitexact
    SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_conv.c
         ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_wsrelu.c
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fxp_opt_bitexact.cpp)

dcvc_onnx_add_executable(bench_fxp_conv
    SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_conv.c
         ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_wsrelu.c
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/fxp_baseline_kernels.c
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_fxp_conv.cpp)
if(NOT MSVC)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fxp_baseline_kernels.c
        PROPERTIES COMPILE_OPTIONS "-O3;-mavx2"
    )
endif()

if(DCVC_FXP_CUDA_ENABLED)
    dcvc_onnx_add_executable(test_fxp_cuda_bitexact
        SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_conv.c
             ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_wsrelu.c
             ${CMAKE_CURRENT_SOURCE_DIR}/src/fxp/fxp_cuda.cu
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fxp_cuda_bitexact.cpp)
endif()

if(DCVC_ORT_GPU AND DCVC_FXP_CUDA_ENABLED)
    dcvc_onnx_add_executable(test_fxp_session_cuda
        SRCS ${DCVC_PIPELINE_SRCS}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fxp_session_cuda.cpp)
endif()

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

dcvc_onnx_add_executable(test_rans_xop
    SRCS ${DCVC_NPY_SRC}
         ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rans_xop.cpp)

# ===========================================================================
# `dcvc_package` target: assemble a ready-to-run folder (binaries + ORT lib
# + models + CDF tables) under out/runnable/<platform>/.
# Works on Linux and Windows without different toolchains clobbering each other.
# ===========================================================================
# Short run instructions dropped into the package root.
file(WRITE "${CMAKE_BINARY_DIR}/README_PACKAGE.txt"
"DCVC CPU ONNX codec - runnable package
======================================
Contents:
  *.exe / test_*   codec test executables
  onnxruntime.dll / libonnxruntime.so*   ONNX Runtime shared library
${_DCVC_MODEL_CONTENTS}

Select the model pack that matches the input network resolution:
${_DCVC_MODEL_QUICK_TESTS}

Optional arguments override the defaults:
  test_cpu_end2end[.exe] [model_dir] [H] [W] [qp]   (geometry must match pack)
  test_ar_cpu[.exe]      [model_dir] [H] [W]        (default 8 8)
")

add_custom_target(dcvc_package
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${DCVC_RUNNABLE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DCVC_RUNNABLE_DIR}"
    # Executables (generator expressions resolve the per-config output path).
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:test_intra_analysis>" "${DCVC_RUNNABLE_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:test_ar_cpu>" "${DCVC_RUNNABLE_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:test_cpu_end2end>" "${DCVC_RUNNABLE_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:test_cpu_inter>" "${DCVC_RUNNABLE_DIR}/"
    # ONNX Runtime shared library (next to the binaries).
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ORT_RUNTIME_LIBS} "${DCVC_RUNNABLE_DIR}/"
    # Discovered/configured model packs + CDF tables.
    ${_DCVC_MODEL_PACKAGE_COMMANDS}
    # Run instructions.
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_BINARY_DIR}/README_PACKAGE.txt" "${DCVC_RUNNABLE_DIR}/"
    DEPENDS test_intra_analysis test_ar_cpu test_cpu_end2end test_cpu_inter
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Assembling runnable package at ${DCVC_RUNNABLE_DIR}"
    VERBATIM
)
