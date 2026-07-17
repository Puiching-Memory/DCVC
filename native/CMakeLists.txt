cmake_minimum_required(VERSION 3.18)
project(dcvc_rt_native C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(DCVC_RT_HAS_TENSORRT "Link NVIDIA TensorRT" OFF)
option(DCVC_RT_HAS_CUDA "Enable CUDA plugins" OFF)
option(DCVC_RT_BUILD_TESTS "Build unit tests" ON)

set(DCVC_RT_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/rans
  ${CMAKE_CURRENT_SOURCE_DIR}/plugins/include
)

# --- librans (C++ core + C API) ---
add_library(dcvc_rans STATIC
  rans/rans.cpp
  rans/rans_c.cpp
  rans/pmf_cdf.cpp
)
target_include_directories(dcvc_rans PUBLIC ${DCVC_RT_INCLUDE})
if(MSVC)
  target_compile_options(dcvc_rans PRIVATE /W3)
else()
  target_compile_options(dcvc_rans PRIVATE -Wall -Wextra)
endif()

# --- plugins (CPU reference only; TRT plugins are compiled standalone) ---
set(DCVC_PLUGIN_SOURCES
  plugins/src/plugins_cpu.c
  plugins/src/ar_prior.c
)
add_library(dcvc_plugins STATIC ${DCVC_PLUGIN_SOURCES})
target_include_directories(dcvc_plugins PUBLIC ${DCVC_RT_INCLUDE})
target_link_libraries(dcvc_plugins PUBLIC dcvc_rans)

# --- core runtime ---
set(DCVC_RT_SOURCES
  src/bitstream.c
  src/color.c
  src/dpb.c
  src/trt_runner.c
  src/dcvc_rt.c
)
if(DCVC_RT_HAS_TENSORRT)
  list(APPEND DCVC_RT_SOURCES src/trt_engine.cpp src/dcvc_ar_codec.c)
  list(APPEND DCVC_RT_SOURCES src/dcvc_intra_pipeline.c)
  list(APPEND DCVC_RT_SOURCES src/dcvc_inter_pipeline.c)
endif()

add_library(dcvc_rt STATIC ${DCVC_RT_SOURCES})
target_include_directories(dcvc_rt PUBLIC ${DCVC_RT_INCLUDE})
target_link_libraries(dcvc_rt PUBLIC dcvc_rans dcvc_plugins)

if(DCVC_RT_HAS_TENSORRT)
  target_compile_definitions(dcvc_rt PUBLIC DCVC_RT_HAS_TENSORRT)
  target_include_directories(dcvc_rt PUBLIC
    /usr/include/x86_64-linux-gnu
    /usr/local/cuda-13.2/include)
  target_link_directories(dcvc_rt PUBLIC /usr/local/cuda-13.2/lib64 /usr/lib/x86_64-linux-gnu)
  target_link_libraries(dcvc_rt PUBLIC nvinfer cudart ${CMAKE_DL_LIBS})
endif()

if(DCVC_RT_HAS_CUDA)
  enable_language(CUDA)
  target_compile_definitions(dcvc_plugins PUBLIC DCVC_RT_HAS_CUDA)
  target_compile_definitions(dcvc_rt PUBLIC DCVC_RT_HAS_CUDA)

  # Host-side CUDA kernels (libdcvc_kernels.so), loaded at runtime via dlopen
  # by the encoder/decoder (dcvc_rt.c). Previously built by a hand-run nvcc
  # command that cmake never tracked -> stale .so / silent dlsym failures.
  # Now a first-class build target: `cmake --build build` rebuilds it.
  add_library(dcvc_kernels SHARED plugins/src/dcvc_kernels.cu)
  target_include_directories(dcvc_kernels PRIVATE ${DCVC_RT_INCLUDE})
  set_target_properties(dcvc_kernels PROPERTIES
    CUDA_ARCHITECTURES "80"
    POSITION_INDEPENDENT_CODE ON)
  if(NOT MSVC)
    target_compile_options(dcvc_kernels PRIVATE -O2)
  endif()
  # Runtime loads it from <build>/plugin_demo/libdcvc_kernels.so
  set_target_properties(dcvc_kernels PROPERTIES
    OUTPUT_NAME "dcvc_kernels"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugin_demo")
endif()

# Threads for rANS multi-thread
find_package(Threads REQUIRED)
target_link_libraries(dcvc_rans PUBLIC Threads::Threads)

if(DCVC_RT_BUILD_TESTS)
  enable_testing()
  add_executable(test_bitstream tests/test_bitstream.c)
  target_link_libraries(test_bitstream PRIVATE dcvc_rt)
  add_test(NAME bitstream COMMAND test_bitstream)

  add_executable(test_rans tests/test_rans.c)
  target_link_libraries(test_rans PRIVATE dcvc_rt)
  add_test(NAME rans COMMAND test_rans)

  add_executable(test_color tests/test_color.c)
  target_link_libraries(test_color PRIVATE dcvc_rt)
  add_test(NAME color COMMAND test_color)

  add_executable(test_roundtrip tests/test_roundtrip.c)
  target_link_libraries(test_roundtrip PRIVATE dcvc_rt)
  add_test(NAME roundtrip COMMAND test_roundtrip)

  add_executable(bench_container tests/bench_container.c)
  target_link_libraries(bench_container PRIVATE dcvc_rt)
endif()

install(TARGETS dcvc_rt dcvc_rans dcvc_plugins ARCHIVE DESTINATION lib)
install(FILES include/dcvc_rt.h DESTINATION include)

# TRT engine integration test
if(DCVC_RT_HAS_TENSORRT AND DCVC_RT_BUILD_TESTS)
  add_executable(test_trt_engine tests/test_trt_engine.c)
  target_link_libraries(test_trt_engine PRIVATE dcvc_rt)
  # test_trt_engine requires CLI args (engine_id input.npy ref.npy); run manually
endif()

# C I-frame decode test
if(DCVC_RT_HAS_TENSORRT AND DCVC_RT_BUILD_TESTS)
  add_executable(test_c_iframe_decode tests/test_c_iframe_decode.c)
  target_link_libraries(test_c_iframe_decode PRIVATE dcvc_rt ${CMAKE_DL_LIBS})
  # test_c_iframe_decode requires CLI args (golden_dir engine_dir plugin_dir cdf_dir); run manually
endif()

# Full C decode pipeline test
if(DCVC_RT_HAS_TENSORRT AND DCVC_RT_BUILD_TESTS)
  add_executable(test_c_decode_full tests/test_c_decode_full.c)
  target_link_libraries(test_c_decode_full PRIVATE dcvc_rt ${CMAKE_DL_LIBS})
  add_test(NAME c_decode_full COMMAND test_c_decode_full WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

# rANS z decode test
if(DCVC_RT_BUILD_TESTS)
  add_executable(test_rans_z_decode tests/test_rans_z_decode.c)
  target_link_libraries(test_rans_z_decode PRIVATE dcvc_rans)
  add_test(NAME rans_z_decode COMMAND test_rans_z_decode WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

# 4x y decode test
if(DCVC_RT_HAS_TENSORRT AND DCVC_RT_BUILD_TESTS)
  add_executable(test_y_decode_4x tests/test_y_decode_4x.c)
  target_link_libraries(test_y_decode_4x PRIVATE dcvc_rt dcvc_rans ${CMAKE_DL_LIBS})
  add_test(NAME y_decode_4x COMMAND test_y_decode_4x WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

# Closed-loop self-consistency test (encode + decode, same TRT engines)
if(DCVC_RT_HAS_TENSORRT AND DCVC_RT_BUILD_TESTS)
  add_executable(test_closed_loop tests/test_closed_loop.c)
  target_link_libraries(test_closed_loop PRIVATE dcvc_rt dcvc_rans ${CMAKE_DL_LIBS})
  add_test(NAME closed_loop COMMAND test_closed_loop WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

# AR codec production module integration test
if(DCVC_RT_HAS_TENSORRT AND DCVC_RT_BUILD_TESTS)
  add_executable(test_ar_codec tests/test_ar_codec.c)
  target_link_libraries(test_ar_codec PRIVATE dcvc_rt ${CMAKE_DL_LIBS})
  add_test(NAME ar_codec COMMAND test_ar_codec WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

# SDK-level I→P→P integration test
if(DCVC_RT_HAS_TENSORRT AND DCVC_RT_BUILD_TESTS)
  add_executable(test_sdk_inter tests/test_sdk_inter.c)
  target_link_libraries(test_sdk_inter PRIVATE dcvc_rt ${CMAKE_DL_LIBS})
  add_test(NAME sdk_inter COMMAND test_sdk_inter WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()
# 2x (inter/P-frame) AR codec round-trip test
if(DCVC_RT_HAS_TENSORRT AND DCVC_RT_BUILD_TESTS)
  add_executable(test_inter_ar tests/test_inter_ar.c)
  target_link_libraries(test_inter_ar PRIVATE dcvc_rt ${CMAKE_DL_LIBS})
  add_test(NAME inter_ar COMMAND test_inter_ar WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

# Full inter (P-frame) pipeline end-to-end closed-loop test
if(DCVC_RT_HAS_TENSORRT AND DCVC_RT_BUILD_TESTS)
  add_executable(test_inter_pipeline tests/test_inter_pipeline.c)
  target_link_libraries(test_inter_pipeline PRIVATE dcvc_rt ${CMAKE_DL_LIBS})
  add_test(NAME inter_pipeline COMMAND test_inter_pipeline WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

# Full intra-frame pipeline end-to-end test
if(DCVC_RT_HAS_TENSORRT AND DCVC_RT_BUILD_TESTS)
  add_executable(test_intra_pipeline tests/test_intra_pipeline.c)
  target_link_libraries(test_intra_pipeline PRIVATE dcvc_rt ${CMAKE_DL_LIBS})
  add_test(NAME intra_pipeline COMMAND test_intra_pipeline WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

# 4x pipeline test (golden symbols)
if(DCVC_RT_HAS_TENSORRT AND DCVC_RT_BUILD_TESTS)
  add_executable(test_4x_pipeline tests/test_4x_pipeline.c)
  target_link_libraries(test_4x_pipeline PRIVATE dcvc_rt ${CMAKE_DL_LIBS})
  add_test(NAME 4x_pipeline COMMAND test_4x_pipeline WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

# 2x AR kernel parity test (inter/P-frame kernels)
if(DCVC_RT_BUILD_TESTS)
  add_executable(test_kernels_2x tests/test_kernels_2x.c)
  target_link_libraries(test_kernels_2x PRIVATE dcvc_rt ${CMAKE_DL_LIBS})
  add_test(NAME kernels_2x COMMAND test_kernels_2x WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

# Codec quality / drift / latency benchmark
if(DCVC_RT_HAS_TENSORRT AND DCVC_RT_BUILD_TESTS)
  add_executable(test_quality tests/test_quality.c)
  target_link_libraries(test_quality PRIVATE dcvc_rt ${CMAKE_DL_LIBS} m)
endif()

# Performance benchmark
if(DCVC_RT_HAS_TENSORRT AND DCVC_RT_BUILD_TESTS)
  add_executable(test_perf tests/test_perf.c)
  target_link_libraries(test_perf PRIVATE dcvc_rt ${CMAKE_DL_LIBS} m)
endif()

