file(GLOB SOURCES "*.cpp" "*.cu")

add_library(tests STATIC ${SOURCES})

target_link_libraries(
  tests
  GTest::gtest_main
  CUDA::cusparse 
  CUDA::cudart
  custatevec
  )