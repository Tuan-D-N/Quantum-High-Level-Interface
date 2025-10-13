#include "ChebyshevMatrixExponentialCoeff.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <span>
#include <optional>
#include <device_launch_parameters.h>
#include "../CudaControl/Helper.hpp"   // CHECK_CUDA / CHECK_CUSPARSE
#include "SparseHelper.hpp"            // gather_kernel / scatter_kernel

// ==========================================================
// Device kernels
// ==========================================================

// out[i] += coeff * tmp[i]
__global__ void axpy_scale_kernel_complex(cuDoubleComplex* out,
                                          const cuDoubleComplex* tmp,
                                          cuDoubleComplex coeff,
                                          int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    cuDoubleComplex term = cuCmul(coeff, tmp[i]);
    out[i] = cuCadd(out[i], term);
}

// out[i] = coeff * in[i]
__global__ void scale_kernel_complex(cuDoubleComplex* out,
                                     const cuDoubleComplex* in,
                                     cuDoubleComplex coeff,
                                     int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = cuCmul(coeff, in[i]);
}


// ==========================================================
// exp(iA) v ≈ Σ_{k=0}^order gamma[k] A^k v     (DEVICE CSR, raw ptrs)
// ==========================================================
int expiAv_chebyshev_gamma_cusparse_device(
    cusparseHandle_t handle,
    int n, int nnz, int order,
    const int* d_csrRowPtr,                  // DEVICE pointers (no span)
    const int* d_csrColInd,                  // DEVICE pointers (no span)
    const cuDoubleComplex* d_csrVal,         // DEVICE pointers (no span)
    const std::vector<cuDoubleComplex> gamma,
    cuDoubleComplex* d_v_in_out)
{
    if (order < 0) return 0;
    if (!d_csrRowPtr || !d_csrColInd || !d_csrVal || !d_v_in_out) return -1;
    if ((int)gamma.size() < order + 1) return -2;

    // cuSPARSE descriptors
    const cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, n, n, nnz,
        (void*)d_csrRowPtr, (void*)d_csrColInd, (void*)d_csrVal,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));

    cuDoubleComplex *d_tmp_in = nullptr, *d_tmp_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_tmp_in,  n * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&d_tmp_out, n * sizeof(cuDoubleComplex)));
    // tmp_in <- v (we keep d_v_in_out as the accumulator/output)
    CHECK_CUDA(cudaMemcpy(d_tmp_in, d_v_in_out, n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

    cusparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, d_tmp_in,  CUDA_C_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, n, d_tmp_out, CUDA_C_64F));

    size_t bufferSize = 0;
    void*  dBuffer = nullptr;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, matA, vecX, &zero, vecY,
        CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    // out = gamma[0] * v
    scale_kernel_complex CUDA_KERNEL(blocks, threads) (d_v_in_out, d_tmp_in, gamma[0], n);
    CHECK_CUDA(cudaGetLastError());

    // Power-series evaluation:
    // tmp_out = A * tmp_in; out += gamma[i] * tmp_out; swap(tmp_in,tmp_out)
    for (int i = 1; i <= order; ++i) {
        CHECK_CUSPARSE(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, matA, vecX, &zero, vecY,
            CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        axpy_scale_kernel_complex CUDA_KERNEL(blocks, threads) (d_v_in_out, d_tmp_out, gamma[i], n);
        CHECK_CUDA(cudaGetLastError());

        std::swap(d_tmp_in, d_tmp_out);
        CHECK_CUSPARSE(cusparseDnVecSetValues(vecX, d_tmp_in));
        CHECK_CUSPARSE(cusparseDnVecSetValues(vecY, d_tmp_out));
    }

    // Cleanup
    cudaFree(dBuffer);
    cudaFree(d_tmp_in);
    cudaFree(d_tmp_out);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matA);
    return 0;
}

// ==========================================================
// exp(iA) v — HOST CSR spans -> (copy CSR to device) -> device eval
// ==========================================================
int expiAv_chebyshev_gamma_cusparse_host(
    cusparseHandle_t handle,
    int n, int nnz, int order,
    std::span<const int> h_csrRowPtr,
    std::span<const int> h_csrColInd,
    std::span<const cuDoubleComplex> h_csrVal,
    cuDoubleComplex* d_v_in_out,
    const double t /* = +1 for exp(-iA), -1 for exp(+iA) */)
{
    if (order < 0) return 0;
    if ((int)h_csrRowPtr.size() != n + 1 || (int)h_csrColInd.size() != nnz || (int)h_csrVal.size() != nnz)
        return -1;

    // gamma from host CSR (spectral-only scaling; auto β = ||A||_1 if not provided)
    std::vector<cuDoubleComplex> gamma =
        chebyshev_exp_gamma_spectral_csr(n, h_csrRowPtr, h_csrColInd, h_csrVal, t, order, std::nullopt);

    // Copy CSR to device
    int *d_r = nullptr, *d_c = nullptr;
    cuDoubleComplex* d_v = nullptr;
    CHECK_CUDA(cudaMalloc(&d_r, sizeof(int)*(n+1)));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(int)*nnz));
    CHECK_CUDA(cudaMalloc(&d_v, sizeof(cuDoubleComplex)*nnz));
    CHECK_CUDA(cudaMemcpy(d_r, h_csrRowPtr.data(), sizeof(int)*(n+1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_csrColInd.data(), sizeof(int)*nnz,  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, h_csrVal.data(),    sizeof(cuDoubleComplex)*nnz, cudaMemcpyHostToDevice));

    // Apply on device
    int rc = expiAv_chebyshev_gamma_cusparse_device(
        handle, n, nnz, order,
        d_r, d_c, d_v,
        gamma,
        d_v_in_out);

    // Free temps
    cudaFree(d_r);
    cudaFree(d_c);
    cudaFree(d_v);
    return rc;
}

// ==========================================================
// Controlled exp(iA) on a target subspace — DEVICE CSR + gamma
// ==========================================================
int applyControlledExpChebyshev_cusparse_device(
    cusparseHandle_t handle,
    int nQubits,
    const int* d_csrRowPtr,                  // DEVICE CSR of the target operator (dim d)
    const int* d_csrColInd,
    const cuDoubleComplex* d_csrVal,
    const std::vector<cuDoubleComplex> gamma,
    cuDoubleComplex* d_state,
    const std::vector<int>& targetQubits,
    const std::vector<int>& controlQubits,
    int nnz,
    int order)
{
    const int k   = static_cast<int>(targetQubits.size());
    const int d   = 1 << k;
    const int nBlocks = 1 << (nQubits - k);

    // Non-target set
    std::vector<int> nonTarget;
    nonTarget.reserve(nQubits - k);
    for (int q = 0; q < nQubits; ++q)
        if (std::find(targetQubits.begin(), targetQubits.end(), q) == targetQubits.end())
            nonTarget.push_back(q);

    // Control mask (|1> assumed)
    int controlMask = 0;
    for (int cq : controlQubits) controlMask |= (1 << cq);

    // Buffers
    cuDoubleComplex* d_block = nullptr;
    CHECK_CUDA(cudaMalloc(&d_block, d * sizeof(cuDoubleComplex)));
    std::vector<int> h_offsets(d);
    int* d_offsets = nullptr;
    CHECK_CUDA(cudaMalloc(&d_offsets, d * sizeof(int)));

    const int threads = 256;
    const int blocks  = (d + threads - 1) / threads;

    for (int blk = 0; blk < nBlocks; ++blk) {
        int nonTargetMask = 0;
        for (size_t i = 0; i < nonTarget.size(); ++i)
            if ((blk >> i) & 1) nonTargetMask |= (1 << nonTarget[i]);

        // Check controls satisfied
        if ((nonTargetMask & controlMask) != controlMask) continue;

        // Build offsets for this block
        for (int b = 0; b < d; ++b) {
            int targetMask = 0;
            for (int q = 0; q < k; ++q)
                if ((b >> q) & 1) targetMask |= (1 << targetQubits[q]);
            h_offsets[b] = nonTargetMask | targetMask;
        }
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), d * sizeof(int), cudaMemcpyHostToDevice));

        // Gather subspace slice
        gather_kernel CUDA_KERNEL(blocks, threads) (d_block, d_state, d_offsets, d);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Apply Σ gamma[i] A^i to the d-dimensional block
        int rc = expiAv_chebyshev_gamma_cusparse_device(
            handle, d, nnz, order,
            d_csrRowPtr, d_csrColInd, d_csrVal,
            gamma,
            d_block);
        if (rc) { cudaFree(d_offsets); cudaFree(d_block); return rc; }

        // Scatter back
        scatter_kernel CUDA_KERNEL(blocks, threads) (d_state, d_block, d_offsets, d);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    cudaFree(d_offsets);
    cudaFree(d_block);
    return 0;
}

// ==========================================================
// Controlled exp(iA) — build gamma from HOST CSR (copied from DEVICE),
// then call device version per-block
// ==========================================================
int applyControlledExpChebyshev_cusparse_host(
    cusparseHandle_t handle,
    int nQubits,
    const int* d_csrRowPtr,                  // DEVICE CSR (dim d)
    const int* d_csrColInd,
    const cuDoubleComplex* d_csrVal,
    cuDoubleComplex* d_state,
    const std::vector<int>& targetQubits,
    const std::vector<int>& controlQubits,
    int nnz,
    int order,
    const double t /* = +1 for exp(-iA), -1 for exp(+iA) */)
{
    const int k = static_cast<int>(targetQubits.size());
    const int d = 1 << k;

    // Pull CSR back to HOST to compute gamma
    std::vector<int> h_r(d + 1), h_c(nnz);
    std::vector<cuDoubleComplex> h_v(nnz);
    CHECK_CUDA(cudaMemcpy(h_r.data(), d_csrRowPtr, sizeof(int)*(d+1), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_csrColInd, sizeof(int)*nnz,  cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_v.data(), d_csrVal,    sizeof(cuDoubleComplex)*nnz, cudaMemcpyDeviceToHost));

    std::vector<cuDoubleComplex> gamma =
        chebyshev_exp_gamma_spectral_csr(
            d,
            std::span<const int>(h_r.data(), d+1),
            std::span<const int>(h_c.data(), nnz),
            std::span<const cuDoubleComplex>(h_v.data(), nnz),
            t, order, std::nullopt);

    // Apply on device block-wise with precomputed gamma
    return applyControlledExpChebyshev_cusparse_device(
        handle,
        nQubits,
        d_csrRowPtr, d_csrColInd, d_csrVal,
        gamma,
        d_state,
        targetQubits, controlQubits,
        nnz,
        order);
}
