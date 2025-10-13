#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <custatevec.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <bitset>
#include "../CudaControl/Helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR
#include <cusparse.h>
#include <vector>
#include <cmath>
#include <span>
#include <optional>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cassert>
#include "Normalise_d.hpp"

// --------- kernels ---------

// Per-block partial reduction of ||v||^2 into d_partials[blockIdx.x]
static __global__ void reduce_norm2_partials_u64(
    const cuDoubleComplex *__restrict__ d_sv,
    unsigned long long len, // length (uint64_t on host)
    double *__restrict__ d_partials)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int TPB = blockDim.x;
    const unsigned long long gid0 = blockIdx.x * static_cast<unsigned long long>(TPB) + tid;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * TPB;

    double acc = 0.0;
    for (unsigned long long i = gid0; i < len; i += stride)
    {
        const cuDoubleComplex z = d_sv[i];
        const double re = cuCreal(z);
        const double im = cuCimag(z);
        acc += re * re + im * im;
    }

    extern __shared__ double smem[]; // size TPB
    smem[tid] = acc;
    __syncthreads();

    // parallel reduction in shared memory
    for (unsigned int s = TPB >> 1u; s > 0u; s >>= 1u)
    {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0u)
        d_partials[blockIdx.x] = smem[0];
}

// Scale vector by factor alpha in place
static __global__ void scale_inplace_u64(
    cuDoubleComplex *__restrict__ d_sv,
    unsigned long long len,
    double alpha)
{
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
    {
        cuDoubleComplex z = d_sv[i];
        d_sv[i] = make_cuDoubleComplex(alpha * cuCreal(z), alpha * cuCimag(z));
    }
}

// --------- host API ---------

/**
 * @brief Normalize a device statevector so that sum |z_i|^2 = 1.
 *
 * @param d_sv       [device] pointer to cuDoubleComplex array
 * @param length     number of elements (uint64_t)
 * @param out_norm2  optional host pointer; if non-null, receives the pre-normalization ||v||^2
 *
 * @return cudaError_t from the last CUDA runtime call (cudaSuccess on success)
 */
cudaError_t square_normalize_statevector_u64(
    cuDoubleComplex *d_sv,
    std::uint64_t length,
    double *out_norm2)
{
    assert(d_sv != nullptr);

    // Trivial cases
    if (length == 0ull)
    {
        if (out_norm2)
            *out_norm2 = 0.0;
        return cudaSuccess;
    }

    // Launch config
    const unsigned int TPB = 256u;
    // cap blocks to something reasonable; you can tune this
    unsigned int blocks = static_cast<unsigned int>((length + TPB - 1ull) / TPB);
    if (blocks == 0u)
        blocks = 1u;
    if (blocks > 65535u)
        blocks = 65535u;

    // Allocate partials (device)
    void *tmp = nullptr;
    cudaError_t st = cudaMalloc(&tmp, sizeof(double) * blocks);
    if (st != cudaSuccess)
        return st;
    double *d_partials = reinterpret_cast<double *>(tmp);

    // 1) partial reductions
    const size_t shmem_bytes = sizeof(double) * TPB;
    reduce_norm2_partials_u64 CUDA_KERNEL(blocks, TPB, shmem_bytes)(d_sv, static_cast<unsigned long long>(length), d_partials);
    st = cudaGetLastError();
    if (st != cudaSuccess)
    {
        cudaFree(d_partials);
        return st;
    }

    // 2) finalize on host
    std::vector<double> h_partials(blocks);
    st = cudaMemcpy(h_partials.data(), d_partials, sizeof(double) * blocks, cudaMemcpyDeviceToHost);
    if (st != cudaSuccess)
    {
        cudaFree(d_partials);
        return st;
    }

    double norm2 = 0.0;
    for (unsigned int i = 0; i < blocks; ++i)
        norm2 += h_partials[i];
    if (out_norm2)
        *out_norm2 = norm2;

    // 3) if zero vector, nothing to scale (leave as-is)
    if (norm2 <= 0.0)
    {
        cudaFree(d_partials);
        return cudaSuccess;
    }

    const double inv_norm = 1.0 / std::sqrt(norm2);

    // 4) scale in place
    const unsigned int grid2 = static_cast<unsigned int>((length + TPB - 1ull) / TPB);
    scale_inplace_u64 CUDA_KERNEL(grid2, TPB)(d_sv, static_cast<unsigned long long>(length), inv_norm);
    st = cudaGetLastError();

    cudaFree(d_partials);
    return st;
}
