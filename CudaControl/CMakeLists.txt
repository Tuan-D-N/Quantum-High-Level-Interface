# Find the CUDA Toolkit package
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_COMPILER "/opt/nvidia/hpc_sdk/Linux_aarch64/24.9/compilers/bin/nvc++")

# Explicitly tell CMake you're using C++ and CUDA by setting languages
project(CudaControl LANGUAGES CXX CUDA)

# Gather all source files (*.cpp, *.cu, *.c)
file(GLOB SOURCES "*.c*" "*.cu")

# Create a static library from the sources
add_library(CudaControl STATIC ${SOURCES})

# Set properties for CUDA compilation
set_target_properties(CudaControl PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)


# Link the necessary CUDA libraries (e.g., cusparse)
target_link_libraries(CudaControl PRIVATE CUDA::cusparse CUDA::cudart radon_functionality)

