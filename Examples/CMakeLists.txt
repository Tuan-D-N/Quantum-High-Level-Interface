# Find the CUDA Toolkit package
find_package(CUDAToolkit REQUIRED)

# Explicitly tell CMake you're using C++ and CUDA by setting languages
project(Examples LANGUAGES CXX CUDA)

# Gather all source files (*.cpp, *.cu, *.c)
file(GLOB SOURCES "*.c*" "*.cu")

# Create a static library from the sources
add_library(Examples STATIC ${SOURCES})

# Set properties for CUDA compilation
set_target_properties(Examples PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link the necessary CUDA libraries (e.g., cusparse)
target_link_libraries(Examples PRIVATE CUDA::cusparse CUDA::cudart)