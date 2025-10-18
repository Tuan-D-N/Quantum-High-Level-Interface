#include "SparseGateBULK.hpp"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <vector>
#include <algorithm>
#include "../CudaControl/Helper.hpp"
#include "SparseHelper.hpp"

// ---- helpers: bulk gather / scatter (column-major matrices: ld = d) ----
__global__ void bulk_gather_cols(
    cuDoubleComplex *__restrict__ M, int ld,      // M is d x B (column-major)
    const cuDoubleComplex *__restrict__ state_in, // full state vector
    const int *__restrict__ offsets,              // size d*B; column j then row r
    int d, int B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d * B;
    if (idx >= total)
        return;
    int col = idx / d;       // 0..B-1
    int row = idx - col * d; // 0..d-1
    int gidx = offsets[col * d + row];
    M[col * ld + row] = state_in[gidx];
}

__global__ void bulk_scatter_cols(
    cuDoubleComplex *__restrict__ state_out,       // full state vector
    const cuDoubleComplex *__restrict__ M, int ld, // Y matrix d x B (column-major)
    const int *__restrict__ offsets,               // size d*B
    int d, int B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d * B;
    if (idx >= total)
        return;
    int col = idx / d;
    int row = idx - col * d;
    int gidx = offsets[col * d + row];
    state_out[gidx] = M[col * ld + row];
}

// Apply sparse gate U to given qubits: optimized with bulk SpMM
int applySparseGateBulk(
    cusparseHandle_t handle,
    int nQubits,
    const int *d_csrRowPtrU,
    const int *d_csrColIndU,
    const cuDoubleComplex *d_csrValU,
    const cuDoubleComplex *d_state_in,
    cuDoubleComplex *d_state_out,
    const std::vector<int> &targetQubits,
    const std::vector<int> &controlQubits,
    int nnzU)
{
    const int k = static_cast<int>(targetQubits.size());
    const int d = 1 << k;
    const int dim = 1 << nQubits;

    // (Optional but robust) pass-through initialization for blocks that won't be touched.
    // If you want exact parity with your scalar path, do the same there or remove this.
    if (d_state_out != d_state_in)
    {
        CHECK_CUDA(cudaMemcpy((void *)d_state_out, (const void *)d_state_in,
                              sizeof(cuDoubleComplex) * dim, cudaMemcpyDeviceToDevice));
    }

    // --- Sparse descriptor for U (d x d) ---
    cusparseSpMatDescr_t matU;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matU, d, d, nnzU,
        (void *)d_csrRowPtrU, (void *)d_csrColIndU, (void *)d_csrValU,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));

    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    // --- Build non-target list & control mask ---
    std::vector<int> nonTargetQubits;
    nonTargetQubits.reserve(nQubits - k);
    for (int q = 0; q < nQubits; ++q)
        if (std::find(targetQubits.begin(), targetQubits.end(), q) == targetQubits.end())
            nonTargetQubits.push_back(q);

    int controlMask = 0;
    for (int cq : controlQubits)
        controlMask |= (1 << cq);

    // --- Enumerate eligible blocks (controls satisfied) ---
    const int nBlocks = 1 << (nQubits - k);
    std::vector<int> workBlocks;
    workBlocks.reserve(nBlocks);
    for (int blk = 0; blk < nBlocks; ++blk)
    {
        int nonTargetMask = 0;
        for (size_t i = 0; i < nonTargetQubits.size(); ++i)
            if ((blk >> i) & 1)
                nonTargetMask |= (1 << nonTargetQubits[i]);
        if ((nonTargetMask & controlMask) == controlMask)
            workBlocks.push_back(blk);
    }
    if (workBlocks.empty())
    {
        cusparseDestroySpMat(matU);
        return 0;
    }

    // --- Buffers / descriptors reused across batches ---
    cuDoubleComplex *d_X = nullptr, *d_Y = nullptr;
    int *d_offsets = nullptr;
    void *dBuffer = nullptr;
    size_t dBufferSize = 0;

    // Dense descriptors (will be recreated if B changes)
    cusparseDnMatDescr_t matX = nullptr, matY = nullptr;
    int currentB = 0; // track current descriptor column count

    auto destroyDnMats = [&]()
    {
        if (matX)
        {
            cusparseDestroyDnMat(matX);
            matX = nullptr;
        }
        if (matY)
        {
            cusparseDestroyDnMat(matY);
            matY = nullptr;
        }
    };

    // Simple soft cap heuristic
    const size_t capBytes = size_t(64) << 20; // ~64 MiB
    auto maxBByMem = [&](size_t wsBytes)
    {
        // per column B: X(d) + Y(d) complexes + offsets(d) ints
        // 16 bytes per cuDoubleComplex, 4 bytes per int
        double perB = double(d) * 16.0 + double(d) * 16.0 + double(d) * 4.0;
        double avail = double(capBytes > wsBytes ? (capBytes - wsBytes) : 0);
        int B = (perB > 0.0) ? int(avail / perB) : 1;
        return std::max(1, B);
    };

    const int totalW = static_cast<int>(workBlocks.size());
    int processed = 0;

    while (processed < totalW)
    {
        // ---- 1) Start with a guess for B ----
        int remaining = totalW - processed;
        int Bguess = std::min(remaining, std::max(1, 64 / std::max(1, d / 64)));

        // Create temp descriptors for query at Bguess (if needed)
        if (!matX || currentB != Bguess)
        {
            destroyDnMats();
            CHECK_CUSPARSE(cusparseCreateDnMat(&matX, d, Bguess, d, nullptr, CUDA_C_64F, CUSPARSE_ORDER_COL));
            CHECK_CUSPARSE(cusparseCreateDnMat(&matY, d, Bguess, d, nullptr, CUDA_C_64F, CUSPARSE_ORDER_COL));
            currentB = Bguess;
        }

        // Query workspace for this Bguess
        size_t needed_guess = 0;
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matU, matX, &beta, matY, CUDA_C_64F,
            CUSPARSE_SPMM_ALG_DEFAULT, &needed_guess));

        // Cap B by memory
        int B = std::min(remaining, std::min(maxBByMem(needed_guess), Bguess));
        if (B < 1)
            B = 1;

        // If B changed from the guess, recreate descriptors and re-query workspace for the FINAL B
        if (B != currentB)
        {
            destroyDnMats();
            CHECK_CUSPARSE(cusparseCreateDnMat(&matX, d, B, d, nullptr, CUDA_C_64F, CUSPARSE_ORDER_COL));
            CHECK_CUSPARSE(cusparseCreateDnMat(&matY, d, B, d, nullptr, CUDA_C_64F, CUSPARSE_ORDER_COL));
            currentB = B;
        }
        size_t needed = 0;
        CHECK_CUSPARSE(cusparseSpMM_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matU, matX, &beta, matY, CUDA_C_64F,
            CUSPARSE_SPMM_ALG_DEFAULT, &needed));

        // Allocate/reallocate for final B
        const size_t bytesX = size_t(d) * B * sizeof(cuDoubleComplex);
        const size_t bytesY = size_t(d) * B * sizeof(cuDoubleComplex);
        const size_t bytesO = size_t(d) * B * sizeof(int);

        auto reallocIfNeeded = [](void **ptr, size_t bytes)
        {
            if (*ptr)
                cudaFree(*ptr);
            CHECK_CUDA(cudaMalloc(ptr, bytes));
            return static_cast<int>(cudaSuccess);
        };
        if (!d_X)
            reallocIfNeeded((void **)&d_X, bytesX);
        else
        {
            cudaFree(d_X);
            CHECK_CUDA(cudaMalloc((void **)&d_X, bytesX));
        }
        if (!d_Y)
            reallocIfNeeded((void **)&d_Y, bytesY);
        else
        {
            cudaFree(d_Y);
            CHECK_CUDA(cudaMalloc((void **)&d_Y, bytesY));
        }
        if (!d_offsets)
            reallocIfNeeded((void **)&d_offsets, bytesO);
        else
        {
            cudaFree(d_offsets);
            CHECK_CUDA(cudaMalloc((void **)&d_offsets, bytesO));
        }

        if (!dBuffer || dBufferSize < needed)
        {
            if (dBuffer)
                cudaFree(dBuffer);
            CHECK_CUDA(cudaMalloc(&dBuffer, needed));
            dBufferSize = needed;
        }

        // Bind values to DnMats
        CHECK_CUSPARSE(cusparseDnMatSetValues(matX, d_X));
        CHECK_CUSPARSE(cusparseDnMatSetValues(matY, d_Y));

        // ---- 2) Build offsets for this batch ----
        std::vector<int> h_offsets(size_t(d) * B);
        for (int j = 0; j < B; ++j)
        {
            int blk = workBlocks[processed + j];
            int nonTargetMask = 0;
            for (size_t i = 0; i < nonTargetQubits.size(); ++i)
                if ((blk >> i) & 1)
                    nonTargetMask |= (1 << nonTargetQubits[i]);
            for (int b = 0; b < d; ++b)
            {
                int targetMask = 0;
                for (int q = 0; q < k; ++q)
                    if ((b >> q) & 1)
                        targetMask |= (1 << targetQubits[q]);
                h_offsets[j * d + b] = nonTargetMask | targetMask; // column-major fill
            }
        }
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), bytesO, cudaMemcpyHostToDevice));

        // ---- 3) Bulk gather X (d x B, column-major) ----
        {
            int threads = 256;
            int blocks = int((size_t(d) * B + threads - 1) / threads);
            bulk_gather_cols CUDA_KERNEL(blocks, threads)(d_X, d, d_state_in, d_offsets, d, B);
            CHECK_CUDA(cudaGetLastError());
        }

        // ---- 4) SpMM: Y = U * X ----
        CHECK_CUSPARSE(cusparseSpMM(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matU, matX, &beta, matY, CUDA_C_64F,
            CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

        // ---- 5) Bulk scatter Y back ----
        {
            int threads = 256;
            int blocks = int((size_t(d) * B + threads - 1) / threads);
            bulk_scatter_cols CUDA_KERNEL(blocks, threads)(d_state_out, d_Y, d, d_offsets, d, B);
            CHECK_CUDA(cudaGetLastError());
        }

        processed += B;
    }

    // Cleanup
    if (d_X)
        cudaFree(d_X);
    if (d_Y)
        cudaFree(d_Y);
    if (d_offsets)
        cudaFree(d_offsets);
    if (dBuffer)
        cudaFree(dBuffer);
    if (matX)
        cusparseDestroyDnMat(matX);
    if (matY)
        cusparseDestroyDnMat(matY);
    cusparseDestroySpMat(matU);
    return 0;
}
