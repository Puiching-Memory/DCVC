cmake_minimum_required(VERSION 3.18)
project(dcvc_rknn C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_compile_options(-Wall -Wextra -O3 -fopenmp -march=armv8-a)

set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")

find_library(RKNNRT_LIB rknnrt
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib
  REQUIRED)
find_path(RKNN_INCLUDE_DIR rknn_api.h
  PATHS /usr/include
  REQUIRED)
message(STATUS "librknnrt: ${RKNNRT_LIB}")
message(STATUS "rknn headers: ${RKNN_INCLUDE_DIR}")

set(RKNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

add_library(dcvc_rk_rans STATIC
  ${RKNN_ROOT}/rans/rans.cpp
  ${RKNN_ROOT}/rans/rans_c.cpp
  ${RKNN_ROOT}/rans/pmf_cdf.cpp)
target_include_directories(dcvc_rk_rans PUBLIC ${RKNN_ROOT}/rans)
target_compile_features(dcvc_rk_rans PUBLIC cxx_std_17)

add_library(dcvc_rk STATIC
  ${RKNN_ROOT}/src/rknn_engine.c
  ${RKNN_ROOT}/src/kernels.c
  ${RKNN_ROOT}/src/npy_reader.c
  ${RKNN_ROOT}/src/fxp_scale_index.c
  ${RKNN_ROOT}/src/ar_codec.c
  ${RKNN_ROOT}/src/intra_pipeline.c
  ${RKNN_ROOT}/src/inter_ld_pipeline.c)
target_include_directories(dcvc_rk
  PUBLIC  ${RKNN_ROOT}/include
  PRIVATE ${RKNN_ROOT}/src ${RKNN_ROOT}/rans ${RKNN_INCLUDE_DIR})
target_link_libraries(dcvc_rk PUBLIC dcvc_rk_rans ${RKNNRT_LIB} m pthread gomp)
set_target_properties(dcvc_rk PROPERTIES
  C_STANDARD 11
  LINKER_LANGUAGE CXX)

add_executable(test_rknn_e2e ${RKNN_ROOT}/tests/test_rknn_e2e.c)
target_link_libraries(test_rknn_e2e PRIVATE dcvc_rk m)
target_include_directories(test_rknn_e2e PRIVATE ${RKNN_ROOT}/include)

add_executable(test_rknn_smoke ${RKNN_ROOT}/tests/test_rknn_smoke.c)
target_link_libraries(test_rknn_smoke PRIVATE dcvc_rk m)
target_include_directories(test_rknn_smoke PRIVATE ${RKNN_ROOT}/include)

add_executable(test_rknn_io_bench ${RKNN_ROOT}/tests/test_rknn_io_bench.c)
target_link_libraries(test_rknn_io_bench PRIVATE dcvc_rk m)
target_include_directories(test_rknn_io_bench PRIVATE ${RKNN_ROOT}/include ${RKNN_INCLUDE_DIR})
