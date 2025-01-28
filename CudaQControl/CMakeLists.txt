# Find the CUDA Toolkit package
find_package(CUDAToolkit REQUIRED)

# set(CMAKE_CXX_COMPILER "/software/ella/software/2024.10/py-3.12.4-cudaq-0.8.0/cudaq/bin/nvq++")

# Explicitly tell CMake you're using C++ and CUDA by setting languages
project(CudaQControl LANGUAGES CXX CUDA)

# Gather all source files (*.cpp, *.cu, *.c)
file(GLOB SOURCES "*.c*" "*.cu")

# Create a static library from the sources
add_library(CudaQControl STATIC ${SOURCES})

# Set properties for CUDA compilation
set_target_properties(CudaQControl PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)


# Link the necessary CUDA libraries (e.g., cusparse)
target_link_libraries(CudaQControl INTERFACE CUDA::cusparse CUDA::cudart radon_functionality)

