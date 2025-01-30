cmake_minimum_required(VERSION 3.22)

project(radon_transform LANGUAGES CXX CUDA)

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)

# Set the C++ compiler
set(CMAKE_CXX_COMPILER "/opt/nvidia/hpc_sdk/Linux_aarch64/24.5/compilers/bin/nvc++")
set(CMAKE_CUDA_COMPILER "/opt/nvidia/hpc_sdk/Linux_aarch64/24.5/compilers/bin/nvcc")

# Set C++ standard
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 --diag_suppress=177 --diag_suppress=550 -gpu=mem:unified")
set(CMAKE_CUDA_FLAGS "-std=c++20 -diag-suppress=550 -diag-suppress=177")

message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")

# Add executable
add_executable(${PROJECT_NAME} main.cpp)

set(CUDA_INClUDE_LIBRARIES "")
set(CUDA_LINK_LIBRARIES "/opt/nvidia/hpc_sdk/Linux_aarch64/24.5/cuda/../math_libs/lib64")
set(CUDA_INClUDE_LIBRARIES "/opt/nvidia/hpc_sdk/Linux_aarch64/24.5/cuda/../math_libs/include")
set(CUDA_INClUDE_LIBRARIES "${CUDA_INClUDE_LIBRARIES}:/opt/nvidia/hpc_sdk/Linux_aarch64/24.5/cuda/12.4/targets/sbsa-linux/include")

link_directories(/software/ella/software/2024.10/py-3.12.4-cudaq-0.8.0/cudaq/lib)

include_directories(/opt/nvidia/hpc_sdk/Linux_aarch64/24.5/cuda/12.4/targets/sbsa-linux/include)
# Include directories
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CUDA_INClUDE_LIBRARIES}
)

message(STATUS "CUDA_INClUDE_LIBRARIES: ${CUDA_INClUDE_LIBRARIES}")

# Link directories
target_link_directories(${PROJECT_NAME} PUBLIC
    ${CUDA_LINK_LIBRARIES}
)

message(STATUS "CUDA_LINK_LIBRARIES: ${CUDA_LINK_LIBRARIES}")

add_subdirectory("functionality")
add_subdirectory("functionality_image")
add_subdirectory("CudaQControl")
add_subdirectory("Examples")
add_subdirectory("CuSparseControl")
add_subdirectory("CuQuantumControl")
add_subdirectory("Runner")

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    CudaQControl
    functionality_image
    radon_functionality
    CuSparseControl
    CUDA::cusparse 
    CUDA::cudart
    custatevec
    CuQuantumControl
    Runner
    ${CUDA_LIBRARIES}
)
set(CMAKE_SKIP_RPATH TRUE)


set(groverBenchmark FALSE)

message(STATUS "groverBenchmark = ${groverBenchmark}")
if(groverBenchmark)
  add_executable(
    groverBenchmarkEXE
    groverBenchmarking.cpp
  )

  target_link_libraries(groverBenchmarkEXE PRIVATE
    CudaQControl
    functionality_image
    CuQuantumControl
    CuSparseControl
    Runner
    radon_functionality
    CUDA::cusparse 
    CUDA::cudart
    custatevec
    )

  target_include_directories(groverBenchmarkEXE PUBLIC
    ${CUDA_INClUDE_LIBRARIES}
  )
  target_link_directories(groverBenchmarkEXE PUBLIC
    ${CUDA_LINK_LIBRARIES}
  )
endif()


set(rawTester TRUE)

message(STATUS "rawTester = ${rawTester}")
if(rawTester)
  add_executable(
    rawTesterEXE
    rawTestering.cu
  )

  target_link_libraries(rawTesterEXE PRIVATE
    CudaQControl
    functionality_image
    CuQuantumControl
    CuSparseControl
    Runner
    radon_functionality
    CUDA::cusparse 
    CUDA::cudart
    custatevec

    cudaq
    cudaq-common
    cudaq-mlir-runtime
     # cudaq-builder  Has memory leaks??????
    cudaq-ensmallen
    cudaq-nlopt
    cudaq-spin
    cudaq-em-default
    cudaq-platform-default
    nvqir
    nvqir-custatevec-fp32
    )

  target_include_directories(rawTesterEXE PUBLIC
    ${CUDA_INClUDE_LIBRARIES}
  )
  target_link_directories(rawTesterEXE PUBLIC
    ${CUDA_LINK_LIBRARIES}
  )
endif()


set(BenchmarkData TRUE)

message(STATUS "BenchmarkData = ${BenchmarkData}")
if(BenchmarkData)
  add_executable(
    BenchmarkDataEXE
    benchmarkingData.cpp
  )

  target_link_libraries(BenchmarkDataEXE PRIVATE
    CudaQControl
    functionality_image
    CuQuantumControl
    CuSparseControl
    Runner
    radon_functionality
    CUDA::cusparse 
    CUDA::cudart
    custatevec

    # cudaq
    # cudaq-common
    # cudaq-mlir-runtime
    # # cudaq-builder  Has memory leaks??????
    # cudaq-ensmallen
    # cudaq-nlopt
    # cudaq-spin
    # cudaq-em-default
    # cudaq-platform-default
    # nvqir
    # nvqir-custatevec-fp32
    )

  target_include_directories(BenchmarkDataEXE PUBLIC
    ${CUDA_INClUDE_LIBRARIES}
  )
  target_link_directories(BenchmarkDataEXE PUBLIC
    ${CUDA_LINK_LIBRARIES}
  )
endif()



set(BUILD_TESTING TRUE)

message(STATUS "Testing = ${BUILD_TESTING}")
if(BUILD_TESTING)

  include(FetchContent)
  FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(googletest)



  add_subdirectory("tests")
  file(GLOB TestSources "tests/*.cpp")

  add_executable(
    runTest
    ${TestSources}
  )

  target_link_libraries(runTest PRIVATE
    CudaQControl
    functionality_image
    GTest::gtest_main
    radon_functionality
    CuQuantumControl
    CuSparseControl
    CUDA::cusparse 
    CUDA::cudart
    custatevec
    )

  target_include_directories(runTest PUBLIC
    ${CUDA_INClUDE_LIBRARIES}
  )
  target_link_directories(runTest PUBLIC
    ${CUDA_LINK_LIBRARIES}
  )

    
  include(GoogleTest)
  gtest_discover_tests(runTest)
endif()

