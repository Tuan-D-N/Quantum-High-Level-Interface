#pragma once
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.

// ===============================
// Gather / Scatter kernels
// ===============================
static __global__ void gather_kernel(cuDoubleComplex *d_out, const cuDoubleComplex *d_in, const unsigned long long *d_offsets, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        d_out[i] = d_in[d_offsets[i]];
}

static __global__ void scatter_kernel(cuDoubleComplex *d_out, const cuDoubleComplex *d_in, const unsigned long long *d_offsets, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        d_out[d_offsets[i]] = d_in[i];
}

static __global__ void gather_kernel(cuDoubleComplex *d_out, const cuDoubleComplex *d_in, const int*d_offsets, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        d_out[i] = d_in[d_offsets[i]];
}

static __global__ void scatter_kernel(cuDoubleComplex *d_out, const cuDoubleComplex *d_in, const int*d_offsets, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        d_out[d_offsets[i]] = d_in[i];
}