file(GLOB SOURCES "*.cpp")

add_library(tests STATIC ${SOURCES})

target_link_libraries(
    tests
  GTest::gtest_main
  )