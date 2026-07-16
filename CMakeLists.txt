cmake_minimum_required(VERSION 3.24)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

project(epq_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(DEFAULT_DEPS_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/local")
set(DEFAULT_FAISS_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/local-avx2")
if(NOT EXISTS "${DEFAULT_FAISS_PREFIX}/share/faiss/faiss-config.cmake")
  set(DEFAULT_FAISS_PREFIX "${DEFAULT_DEPS_PREFIX}")
endif()

set(FAISS_PREFIX "${DEFAULT_FAISS_PREFIX}" CACHE PATH "Faiss install prefix")
set(DEPS_PREFIX "${DEFAULT_DEPS_PREFIX}" CACHE PATH "Eigen3/nlohmann_json install prefix")
set(EPQ_FAISS_TARGET "" CACHE STRING "Faiss target to link: auto, faiss, or faiss_avx2")
set(EPQ_STRUCTURE_DEBUG_LEVEL "0" CACHE STRING "Compile-time structure debug level: 0=off, 1=stage, 2=iteration")
option(EPQ_ENABLE_STRUCTURE_TRACE "Enable optional structure-search trace output" OFF)
option(EPQ_ENABLE_AVQ "Enable AVQ adapter via official ScaNN Python bindings" ON)
set(VAQ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/VAQ" CACHE PATH "TheDatumOrg/VAQ source tree")
set(VAQ_DEPS_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/local-vaq" CACHE PATH "Optional GLPK/Armadillo dependency prefix")

set(faiss_DIR "${FAISS_PREFIX}/share/faiss" CACHE PATH "Path to faiss-config.cmake")
set(Eigen3_DIR "${DEPS_PREFIX}/share/eigen3/cmake" CACHE PATH "Path to Eigen3Config.cmake")
set(nlohmann_json_DIR "${DEPS_PREFIX}/share/cmake/nlohmann_json" CACHE PATH "Path to nlohmann_jsonConfig.cmake")

find_package(OpenMP REQUIRED)
find_package(faiss CONFIG REQUIRED NO_DEFAULT_PATH)
find_package(Eigen3 CONFIG REQUIRED NO_DEFAULT_PATH)
find_package(nlohmann_json CONFIG REQUIRED NO_DEFAULT_PATH)
if(EPQ_ENABLE_AVQ)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
endif()

if(EPQ_FAISS_TARGET STREQUAL "")
  if(TARGET faiss_avx2)
    set(_epq_faiss_target faiss_avx2)
  else()
    set(_epq_faiss_target faiss)
  endif()
else()
  if(NOT TARGET "${EPQ_FAISS_TARGET}")
    message(FATAL_ERROR "Requested EPQ_FAISS_TARGET='${EPQ_FAISS_TARGET}', but that imported target was not found")
  endif()
  set(_epq_faiss_target "${EPQ_FAISS_TARGET}")
endif()

if(NOT EPQ_STRUCTURE_DEBUG_LEVEL MATCHES "^[0-9]+$")
  message(FATAL_ERROR "EPQ_STRUCTURE_DEBUG_LEVEL must be a non-negative integer")
endif()

message(STATUS "Using Faiss target: ${_epq_faiss_target}")
message(STATUS "Structure debug level: ${EPQ_STRUCTURE_DEBUG_LEVEL}")
message(STATUS "Structure trace: ${EPQ_ENABLE_STRUCTURE_TRACE}")

find_path(VAQ_GLPK_INCLUDE_DIR
  NAMES glpk.h
  HINTS "${VAQ_DEPS_PREFIX}/usr/include" "${VAQ_DEPS_PREFIX}/include"
)
find_library(VAQ_GLPK_LIBRARY
  NAMES glpk
  HINTS "${VAQ_DEPS_PREFIX}/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" "${VAQ_DEPS_PREFIX}/usr/lib" "${VAQ_DEPS_PREFIX}/lib"
)
find_path(VAQ_ARMADILLO_INCLUDE_DIR
  NAMES armadillo
  HINTS "${VAQ_DEPS_PREFIX}/usr/include" "${VAQ_DEPS_PREFIX}/include"
)
find_library(VAQ_ARMADILLO_LIBRARY
  NAMES armadillo
  HINTS "${VAQ_DEPS_PREFIX}/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" "${VAQ_DEPS_PREFIX}/usr/lib" "${VAQ_DEPS_PREFIX}/lib"
)
find_library(VAQ_AMD_LIBRARY
  NAMES amd libamd.so.3
  HINTS "${VAQ_DEPS_PREFIX}/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" "${VAQ_DEPS_PREFIX}/usr/lib" "${VAQ_DEPS_PREFIX}/lib"
)
find_library(VAQ_COLAMD_LIBRARY
  NAMES colamd libcolamd.so.3
  HINTS "${VAQ_DEPS_PREFIX}/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" "${VAQ_DEPS_PREFIX}/usr/lib" "${VAQ_DEPS_PREFIX}/lib"
)
find_library(VAQ_SUITESPARSECONFIG_LIBRARY
  NAMES suitesparseconfig libsuitesparseconfig.so.7
  HINTS "${VAQ_DEPS_PREFIX}/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" "${VAQ_DEPS_PREFIX}/usr/lib" "${VAQ_DEPS_PREFIX}/lib"
)
if(NOT VAQ_GLPK_INCLUDE_DIR OR NOT VAQ_GLPK_LIBRARY OR
   NOT VAQ_ARMADILLO_INCLUDE_DIR OR NOT VAQ_ARMADILLO_LIBRARY)
  message(FATAL_ERROR
    "VAQ requires GLPK and Armadillo development files. Install libglpk-dev and libarmadillo-dev, "
    "or set VAQ_DEPS_PREFIX to a prefix containing their headers and libraries.")
endif()

# Generate an auditable compatibility copy while leaving the third-party
# gitlink untouched. The replacements fix upstream sampling, use the proper
# symmetric eigensolver, and avoid an invalid Eigen row/column assignment.
set(_vaq_upstream_source "${VAQ_ROOT}/bitvecengine/VAQ.cpp")
file(READ "${_vaq_upstream_source}" _vaq_source)
set(_vaq_sample_guard "if (sampleSize == XTrain.rows()) {")
string(FIND "${_vaq_source}" "${_vaq_sample_guard}" _vaq_sample_guard_pos)
if(_vaq_sample_guard_pos EQUAL -1)
  message(FATAL_ERROR "Unsupported VAQ source revision: k-means sampling guard was not found")
endif()
string(REPLACE
  "${_vaq_sample_guard}"
  "{ // EPQ integration: sample for both full and partial training slices."
  _vaq_patched_source
  "${_vaq_source}"
)
set(_vaq_eigensolver "Eigen::EigenSolver<RowMatrixXf> es(covmat);")
string(FIND "${_vaq_patched_source}" "${_vaq_eigensolver}" _vaq_eigensolver_pos)
if(_vaq_eigensolver_pos EQUAL -1)
  message(FATAL_ERROR "Unsupported VAQ source revision: covariance eigensolver was not found")
endif()
string(REPLACE
  "${_vaq_eigensolver}"
  "Eigen::SelfAdjointEigenSolver<RowMatrixXf> es(covmat);"
  _vaq_patched_source
  "${_vaq_patched_source}"
)
set(_vaq_column_swap
  "RowVector<Eigen::scomplex> tempc = mEigenVectors.col(i);\n    mEigenVectors.col(i) = mEigenVectors.col(i*mSubsLen + (mSubsLen-1));\n    mEigenVectors.col(i*mSubsLen + (mSubsLen-1)) = tempc;")
string(FIND "${_vaq_patched_source}" "${_vaq_column_swap}" _vaq_column_swap_pos)
if(_vaq_column_swap_pos EQUAL -1)
  message(FATAL_ERROR "Unsupported VAQ source revision: eigenvector column swap was not found")
endif()
string(REPLACE
  "${_vaq_column_swap}"
  "for (int row = 0; row < mEigenVectors.rows(); ++row) {\n      std::swap(mEigenVectors(row, i), mEigenVectors(row, i*mSubsLen + (mSubsLen-1)));\n    }"
  _vaq_patched_source
  "${_vaq_patched_source}"
)
set(_vaq_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/vaq")
file(MAKE_DIRECTORY "${_vaq_generated_dir}")
file(WRITE "${_vaq_generated_dir}/VAQ.cpp" "${_vaq_patched_source}")
file(READ "${VAQ_ROOT}/bitvecengine/VAQ.hpp" _vaq_header)
set(_vaq_blas_guard "extern \"C\" {\n#ifndef FINTEGER")
string(FIND "${_vaq_header}" "${_vaq_blas_guard}" _vaq_blas_guard_pos)
if(_vaq_blas_guard_pos EQUAL -1)
  message(FATAL_ERROR "Unsupported VAQ source revision: BLAS declarations were not found")
endif()
string(REPLACE
  "${_vaq_blas_guard}"
  "#ifndef EIGEN_USE_BLAS\nextern \"C\" {\n#ifndef FINTEGER"
  _vaq_patched_header
  "${_vaq_header}"
)
set(_vaq_class_boundary "}\n\nclass VAQ {")
string(FIND "${_vaq_patched_header}" "${_vaq_class_boundary}" _vaq_class_boundary_pos)
if(_vaq_class_boundary_pos EQUAL -1)
  message(FATAL_ERROR "Unsupported VAQ source revision: VAQ class boundary was not found")
endif()
string(REPLACE
  "${_vaq_class_boundary}"
  "}\n#endif\n\nclass VAQ {"
  _vaq_patched_header
  "${_vaq_patched_header}"
)
file(WRITE "${_vaq_generated_dir}/VAQ.hpp" "${_vaq_patched_header}")
file(WRITE "${_vaq_generated_dir}/VAQIntegration.cpp"
  "#include \"VAQ.cpp\"\n#include \"${CMAKE_CURRENT_SOURCE_DIR}/src/index_vaq.cpp\"\n")

add_library(vaq_upstream STATIC
  "${_vaq_generated_dir}/VAQIntegration.cpp"
  "${VAQ_ROOT}/bitvecengine/utils/Heap.cpp"
  "${VAQ_ROOT}/bitvecengine/utils/FPGrowth/fptree.cpp"
)
target_include_directories(vaq_upstream SYSTEM BEFORE PUBLIC
  "${_vaq_generated_dir}"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${FAISS_PREFIX}/include"
  "${VAQ_ROOT}/bitvecengine"
  "${VAQ_GLPK_INCLUDE_DIR}"
  "${VAQ_ARMADILLO_INCLUDE_DIR}"
)
target_link_libraries(vaq_upstream PUBLIC
  ${_epq_faiss_target}
  Eigen3::Eigen
  nlohmann_json::nlohmann_json
  OpenMP::OpenMP_CXX
  "${VAQ_GLPK_LIBRARY}"
  "${VAQ_ARMADILLO_LIBRARY}"
  ${VAQ_AMD_LIBRARY}
  ${VAQ_COLAMD_LIBRARY}
  ${VAQ_SUITESPARSECONFIG_LIBRARY}
)
target_compile_definitions(vaq_upstream PRIVATE
  EIGEN_DONT_PARALLELIZE
  EIGEN_DONT_ALIGN
  EIGEN_DONT_VECTORIZE
  EIGEN_USE_BLAS
  FINTEGER=int
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")
  target_compile_options(vaq_upstream PRIVATE -mavx2 -mfma)
endif()

add_library(epq
  src/benchmark_ivf_wrappers.cpp
  src/structure.cpp
  src/structure_builder.cpp
  src/structure_builder_grow.cpp
  src/structure_builder_crystallize.cpp
  src/structure_builder_mbeam.cpp
  src/structure_builder_greedy_tail.cpp
  src/structure_builder_chain_tail.cpp
  src/training_config.cpp
  src/serialization_size.cpp
  src/index_arepq.cpp
  src/index_avq.cpp
  src/index_bapq.cpp
  src/index_dpopq.cpp
  src/index_epq.cpp
  src/index_ivf_codec.cpp
)

target_include_directories(epq PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${FAISS_PREFIX}/include
)

target_link_libraries(epq PUBLIC
  ${_epq_faiss_target}
  Eigen3::Eigen
  nlohmann_json::nlohmann_json
  OpenMP::OpenMP_CXX
  vaq_upstream
)

if(EPQ_ENABLE_AVQ)
  target_link_libraries(epq PUBLIC Python3::Python)
endif()

target_compile_definitions(epq PUBLIC
  EPQ_STRUCTURE_DEBUG_LEVEL=${EPQ_STRUCTURE_DEBUG_LEVEL}
  EPQ_ENABLE_STRUCTURE_TRACE=$<BOOL:${EPQ_ENABLE_STRUCTURE_TRACE}>
  EPQ_ENABLE_AVQ=$<BOOL:${EPQ_ENABLE_AVQ}>
  EPQ_ENABLE_VAQ=1
  EPQ_CMAKE_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\"
  EPQ_FAISS_TARGET_NAME=\"${_epq_faiss_target}\"
  EPQ_CMAKE_CXX_COMPILER_ID=\"${CMAKE_CXX_COMPILER_ID}\"
  EPQ_CMAKE_CXX_COMPILER_VERSION=\"${CMAKE_CXX_COMPILER_VERSION}\"
)

add_executable(faiss_smoke_test
  src/faiss_smoke_test.cpp
)

target_include_directories(faiss_smoke_test PRIVATE
  ${FAISS_PREFIX}/include
)

target_link_libraries(faiss_smoke_test PRIVATE
  ${_epq_faiss_target}
  OpenMP::OpenMP_CXX
)

add_executable(epq_stack_smoke_test
  src/epq_stack_smoke_test.cpp
)

target_link_libraries(epq_stack_smoke_test PRIVATE
  epq
)

add_executable(epq_index_smoke_test
  src/epq_index_smoke_test.cpp
)

target_link_libraries(epq_index_smoke_test PRIVATE
  epq
)

add_executable(vaq_bit_packing_smoke_test
  src/vaq_bit_packing_smoke_test.cpp
)

target_include_directories(vaq_bit_packing_smoke_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(epq_lut_smoke_test
  src/epq_lut_smoke_test.cpp
)

target_link_libraries(epq_lut_smoke_test PRIVATE
  epq
)

add_executable(epq_ivf_lut_smoke_test
  src/epq_ivf_lut_smoke_test.cpp
)

target_link_libraries(epq_ivf_lut_smoke_test PRIVATE
  epq
)

add_executable(structure_builder_proxy_smoke_test
  src/structure_builder_proxy_smoke_test.cpp
)

target_link_libraries(structure_builder_proxy_smoke_test PRIVATE
  epq
)

add_executable(architecture_control_builder
  src/architecture_control_builder.cpp
)

target_link_libraries(architecture_control_builder PRIVATE
  epq
)

add_executable(bapq_smoke_test
  src/bapq_smoke_test.cpp
)

target_link_libraries(bapq_smoke_test PRIVATE
  epq
)

add_executable(avq_smoke_test
  src/avq_smoke_test.cpp
)

target_link_libraries(avq_smoke_test PRIVATE
  epq
)

add_executable(flat_benchmark
  src/flat_benchmark.cpp
)

target_link_libraries(flat_benchmark PRIVATE
  epq
)

add_executable(mmeb_vector_benchmark
  src/mmeb_vector_benchmark.cpp
)

target_link_libraries(mmeb_vector_benchmark PRIVATE
  epq
)

add_executable(kmeans_microbench
  src/kmeans_microbench.cpp
)

target_include_directories(kmeans_microbench PRIVATE
  ${FAISS_PREFIX}/include
)

target_link_libraries(kmeans_microbench PRIVATE
  ${_epq_faiss_target}
  OpenMP::OpenMP_CXX
)

add_executable(joint_benchmark
  src/joint_benchmark.cpp
)

target_link_libraries(joint_benchmark PRIVATE
  epq
)
