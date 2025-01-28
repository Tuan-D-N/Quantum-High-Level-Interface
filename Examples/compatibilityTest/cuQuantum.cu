#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void helloWorldKernel() {
    printf("Hello from the GPU! Thread: %d, Block: %d\n", threadIdx.x, blockIdx.x);
}

void run2() {
    // Launching kernel with 1 block and 10 threads
    helloWorldKernel<<<1, 10>>>();

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    std::cout << "Hello from the CPU!" << std::endl;

}
