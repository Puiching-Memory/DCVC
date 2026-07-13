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

# --- plugins (CPU reference + optional TRT) ---
set(DCVC_PLUGIN_SOURCES
  plugins/src/plugins_cpu.c
  plugins/src/ar_prior.c
)
if(DCVC_RT_HAS_TENSORRT)
  list(APPEND DCVC_PLUGIN_SOURCES plugins/src/depthconv_plugin.cpp plugins/src/mask_plugin.cpp)
endif()

add_library(dcvc_plugins STATIC ${DCVC_PLUGIN_SOURCES})
target_include_directories(dcvc_plugins PUBLIC ${DCVC_RT_INCLUDE})
target_link_libraries(dcvc_plugins PUBLIC dcvc_rans)

# --- core runtime ---
add_library(dcvc_rt STATIC
  src/bitstream.c
  src/color.c
  src/dpb.c
  src/trt_runner.c
  src/dcvc_rt.c
)
target_include_directories(dcvc_rt PUBLIC ${DCVC_RT_INCLUDE})
target_link_libraries(dcvc_rt PUBLIC dcvc_rans dcvc_plugins)

if(DCVC_RT_HAS_TENSORRT)
  target_compile_definitions(dcvc_rt PUBLIC DCVC_RT_HAS_TENSORRT)
  target_compile_definitions(dcvc_plugins PUBLIC DCVC_RT_HAS_TENSORRT)
  # User must set TensorRT_ROOT
  if(DEFINED TensorRT_ROOT)
    target_include_directories(dcvc_rt PUBLIC ${TensorRT_ROOT}/include)
    target_include_directories(dcvc_plugins PUBLIC ${TensorRT_ROOT}/include)
    target_link_directories(dcvc_rt PUBLIC ${TensorRT_ROOT}/lib)
    target_link_libraries(dcvc_rt PUBLIC nvinfer)
  endif()
endif()

if(DCVC_RT_HAS_CUDA)
  enable_language(CUDA)
  target_compile_definitions(dcvc_plugins PUBLIC DCVC_RT_HAS_CUDA)
  target_compile_definitions(dcvc_rt PUBLIC DCVC_RT_HAS_CUDA)
  target_sources(dcvc_plugins PRIVATE plugins/src/cuda_kernels.cu)
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
