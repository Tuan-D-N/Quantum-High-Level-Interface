
#include <cmath>

#include "CudaTranspose.hpp"
#include "Helper.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <c++/13/bits/specfun.h>

#define TILE_SIZE 16 // Tile size for shared memory

__global__ void transposeInPlace(float *matrix, int N)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 prevents shared memory bank conflicts

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < N && y < N)
    {
        tile[threadIdx.y][threadIdx.x] = matrix[y * N + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < N && y < N)
    {
        matrix[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose(float *d_matrix, int N)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    transposeInPlace CUDA_KERNEL(grid, block)(d_matrix, N);
    cudaDeviceSynchronize();
}
