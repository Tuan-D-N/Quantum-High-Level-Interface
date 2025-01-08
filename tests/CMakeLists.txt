# Find the CUDA Toolkit package
find_package(CUDAToolkit REQUIRED)

# Explicitly tell CMake you're using C++ and CUDA by setting languages
project(tests LANGUAGES CXX CUDA)

file(GLOB SOURCES "*.cpp" "*.cu")

add_library(tests STATIC ${SOURCES})

target_link_libraries(
  tests
  GTest::gtest_main
  CUDA::cusparse 
  CUDA::cudart
  custatevec
  )