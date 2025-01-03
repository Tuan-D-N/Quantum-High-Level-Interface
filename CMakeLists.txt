cmake_minimum_required(VERSION 3.22)



project(radon_transform LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)


find_program(CLANGXX_PATH clang++)

set(CMAKE_CXX_COMPILER "${CLANGXX_PATH}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/nvidia/cudaq/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D CUDAQ_LIBRARY_MODE -D CUDAQ_SIMULATION_SCALAR_FP64")

# set(CMAKE_CXX_COMPILER nvcc)

message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")

add_executable(${PROJECT_NAME} main.cpp)



add_subdirectory("functionality")

target_link_libraries(${PROJECT_NAME} radon_functionality CUDA::cusparse)




add_subdirectory("tests")

enable_testing()

set(BUILD_TESTING True)

message(STATUS "Testing = ${BUILD_TESTING}")
if(BUILD_TESTING)
  file(GLOB TestSources "tests/*.cpp")

  add_executable(
    runTest
    ${TestSources}
  )

  target_link_libraries(
    runTest
    GTest::gtest_main
    radon_functionality
    )
    
  include(GoogleTest)
  gtest_discover_tests(runTest)
endif()