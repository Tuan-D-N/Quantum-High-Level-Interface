# Find the CUDA Toolkit package
find_package(CUDAToolkit REQUIRED)


# Explicitly tell CMake you're using C++ and CUDA by setting languages
project(CuSparseControl LANGUAGES CXX CUDA)

# Gather all source files (*.cpp, *.cu, *.c)
file(GLOB SOURCES "*.c*" "*.cu")

# Create a static library from the sources
add_library(CuSparseControl STATIC ${SOURCES})

# Set properties for CUDA compilation
set_target_properties(CuSparseControl PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)


# Link the necessary CUDA libraries (e.g., cusparse)
target_link_libraries(CuSparseControl PRIVATE CUDA::cusparse CUDA::cudart radon_functionality CuQuantumControl)

