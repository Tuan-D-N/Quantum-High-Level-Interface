#include <iostream>
#include <math.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <device_launch_parameters.h> //blockDim, blockIdx
#include <cuda_runtime.h> //blockDim, blockIdx

#ifdef __CUDACC__
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#else
#define CUDA_KERNEL(...) 
#endif

// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
 
int main(void)
{
  int N = 1<<20;
  float *x, *y;
 
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
 
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
 
  // Launch kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add CUDA_KERNEL(numBlocks, blockSize) (N, x, y);
 
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
 
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
 
  // Free memory
  cudaFree(x);
  cudaFree(y);
 
  return 0;
}