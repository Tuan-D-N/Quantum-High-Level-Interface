cmake_minimum_required(VERSION 3.22)

project(radon_transform LANGUAGES CXX)

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Set the C++ compiler
set(CMAKE_CXX_COMPILER "/opt/nvidia/hpc_sdk/Linux_aarch64/24.5/compilers/bin/nvc++")
set(CMAKE_CUDA_COMPILER "/opt/nvidia/hpc_sdk/Linux_aarch64/24.5/compilers/bin/nvc++")

# Set C++ standard
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set(CMAKE_CUDA_FLAGS "-std=c++20")

message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")

# Add executable
add_executable(${PROJECT_NAME} main.cpp)

set(CUDA_INClUDE_LIBRARIES "")
set(CUDA_LINK_LIBRARIES "/opt/nvidia/hpc_sdk/Linux_aarch64/24.5/cuda/../math_libs/lib64")
set(CUDA_INClUDE_LIBRARIES "/opt/nvidia/hpc_sdk/Linux_aarch64/24.5/cuda/../math_libs/include")
set(CUDA_INClUDE_LIBRARIES "${CUDA_INClUDE_LIBRARIES}:/opt/nvidia/hpc_sdk/Linux_aarch64/24.5/cuda/12.4/targets/sbsa-linux/include")

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
add_subdirectory("Examples")
add_subdirectory("CuSparseControl")
add_subdirectory("CuQuantumControl")
add_subdirectory("Runner")

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    radon_functionality
    CuSparseControl
    CUDA::cusparse 
    CUDA::cudart
    custatevec
    CuQuantumControl
    Runner
)



set(CMAKE_SKIP_RPATH TRUE)





set(BUILD_TESTING TRUE)

message(STATUS "Testing = ${BUILD_TESTING}")
if(BUILD_TESTING)

  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  FetchContent_MakeAvailable(googletest)



  add_subdirectory("tests")
  file(GLOB TestSources "tests/*.cpp")

  add_executable(
    runTest
    ${TestSources}
  )

  target_link_libraries(
    runTest
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