#include "SparseGate.hpp"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <vector>
#include <algorithm>
#include "../CudaControl/Helper.hpp"
#include "SparseHelper.hpp"

// Apply sparse gate U to given qubits (q0, q1, ..., qk).
// - nQubits: total system qubits
// - d_csrRowPtrU, d_csrColIndU, d_csrValU: CSR of U (size d x d, where d=2^k)
// - d_state_in/out: device statevectors, length 2^n
// - targetQubits: list of k target qubits (ascending order)
// Apply sparse gate U to given qubits (q0, q1, ..., qk)
int applySparseGate(
    cusparseHandle_t handle,
    int nQubits,
    const int *d_csrRowPtrU,
    const int *d_csrColIndU,
    const cuDoubleComplex *d_csrValU,
    const cuDoubleComplex *d_state_in,
    cuDoubleComplex *d_state_out,
    const std::vector<int> &targetQubits,
    int nnzU)
{
    // k is the number of target qubits (dimension of the gate U)
    int k = targetQubits.size();
    // d is the dimension of the sub-block/gate U (d = 2^k)
    int d = 1 << k;
    // dim is the total size of the state vector (dim = 2^nQubits)
    int dim = 1 << nQubits;

    // --- 1. Setup cuSPARSE Matrix Descriptor for the Gate U ---
    cusparseSpMatDescr_t matU;
    CHECK_CUSPARSE(cusparseCreateCsr(&matU, d, d, nnzU, (void *)d_csrRowPtrU, (void *)d_csrColIndU, (void *)d_csrValU,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));

    // Define SpMV scalars: alpha=1.0, beta=0.0 (Operation: Y = 1.0 * U * X + 0.0 * Y)
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    // --- 2. Allocate Scratchpad Memory for Sub-Blocks ---
    cuDoubleComplex *d_block_in;
    cuDoubleComplex *d_block_out;
    // Allocate device memory for the input and output sub-vectors (size d)
    CHECK_CUDA(cudaMalloc(&d_block_in, d * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&d_block_out, d * sizeof(cuDoubleComplex)));

    // --- 3. Setup cuSPARSE Dense Vector Descriptors ---
    cusparseDnVecDescr_t vecX, vecY;
    // Descriptor for input sub-vector X
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, d, d_block_in, CUDA_C_64F));
    // Descriptor for output sub-vector Y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, d, d_block_out, CUDA_C_64F));

    // --- 4. Query and Allocate cuSPARSE Workspace Buffer ---
    size_t bufferSize = 0;
    // Determine the required buffer size for the SpMV operation
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matU, vecX, &beta, vecY,
                                           CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    void *dBuffer = nullptr;
    // Allocate the buffer
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // --- 5. Prepare for Block Iteration ---
    // nBlocks is the number of sub-blocks (size d) the state vector is partitioned into (2^(n-k))
    int nBlocks = 1 << (nQubits - k);

    // Identify qubits NOT in the target list (these define the block index 'blk')
    std::vector<int> nonTargetQubits;
    for (int q = 0; q < nQubits; ++q)
    {
        if (std::find(targetQubits.begin(), targetQubits.end(), q) == targetQubits.end())
        {
            nonTargetQubits.push_back(q);
        }
    }

    // Allocate host and device memory for the state vector offsets
    std::vector<int> h_offsets(d);
    int *d_offsets;
    CHECK_CUDA(cudaMalloc(&d_offsets, d * sizeof(int)));

    // --- 6. Iterate through all computational blocks (Non-Target Qubit States) ---
    for (int blk = 0; blk < nBlocks; ++blk)
    {
        // --- A. Calculate the base offset/mask for the current block ---
        int nonTargetMask = 0;
        // The 'blk' loop index represents the state of the non-target qubits
        for (size_t i = 0; i < nonTargetQubits.size(); ++i)
        {
            if ((blk >> i) & 1)
            {
                nonTargetMask |= (1 << nonTargetQubits[i]);
            }
        }

        // --- B. Calculate full state vector indices for the current block ---
        // Iterate through all 'd' indices within the current block
        for (int b = 0; b < d; ++b)
        {
            int targetMask = 0;
            // The 'b' index represents the state of the target qubits
            for (int q = 0; q < k; ++q)
            {
                if ((b >> q) & 1)
                {
                    targetMask |= (1 << targetQubits[q]);
                }
            }
            // The full index is the sum of the non-target base mask and the target mask
            h_offsets[b] = nonTargetMask | targetMask;
        }

        // 1. Transfer offsets to device
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), d * sizeof(int), cudaMemcpyHostToDevice));

        // 2. Gather: Read input sub-vector from global state into scratchpad
        int threads_per_block = 256;
        int blocks = (d + threads_per_block - 1) / threads_per_block;
        gather_kernel CUDA_KERNEL(blocks, threads_per_block)(d_block_in, d_state_in, d_offsets, d);
        CHECK_CUDA(cudaPeekAtLastError());
        cudaDeviceSynchronize();

        // 3. Apply: Perform the Sparse Matrix-Vector Multiplication (SpMV)
        CHECK_CUDA(cudaMemset(d_block_out, 0, d * sizeof(cuDoubleComplex))); // Clear output block
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matU, vecX, &beta, vecY,
                                    CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());

        // 4. Scatter: Write output sub-vector from scratchpad back to global state
        scatter_kernel CUDA_KERNEL(blocks, threads_per_block)(d_state_out, d_block_out, d_offsets, d);
        CHECK_CUDA(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }

    // --- 7. Cleanup Resources ---
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_block_in));
    CHECK_CUDA(cudaFree(d_block_out));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroySpMat(matU));

    return 0;
}

// Apply sparse gate U to given qubits (q0, q1, ..., qk).
// - nQubits: total system qubits
// - d_csrRowPtrU, d_csrColIndU, d_csrValU: CSR of U (size d x d, where d=2^k)
// - d_state_in/out: device statevectors, length 2^n
// - targetQubits: list of k target qubits (ascending order)
// - controlQubits: list of k control qubits (ascending order)
// Apply sparse gate U to given qubits (q0, q1, ..., qk)
int applySparseGate(
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
    // k is the number of target qubits (dimension of the gate U)
    int k = targetQubits.size();
    // d is the dimension of the sub-block/gate U (d = 2^k)
    int d = 1 << k;
    // dim is the total size of the state vector (dim = 2^nQubits)
    int dim = 1 << nQubits;

    // --- 1. Setup cuSPARSE Matrix Descriptor for the Gate U ---
    cusparseSpMatDescr_t matU;
    CHECK_CUSPARSE(cusparseCreateCsr(&matU, d, d, nnzU, (void *)d_csrRowPtrU, (void *)d_csrColIndU, (void *)d_csrValU,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));

    // Define SpMV scalars: alpha=1.0, beta=0.0 (Operation: Y = 1.0 * U * X + 0.0 * Y)
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    // --- 2. Allocate Scratchpad Memory for Sub-Blocks ---
    cuDoubleComplex *d_block_in;
    cuDoubleComplex *d_block_out;
    // Allocate device memory for the input and output sub-vectors (size d)
    CHECK_CUDA(cudaMalloc(&d_block_in, d * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&d_block_out, d * sizeof(cuDoubleComplex)));

    // --- 3. Setup cuSPARSE Dense Vector Descriptors ---
    cusparseDnVecDescr_t vecX, vecY;
    // Descriptor for input sub-vector X
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, d, d_block_in, CUDA_C_64F));
    // Descriptor for output sub-vector Y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, d, d_block_out, CUDA_C_64F));

    // --- 4. Query and Allocate cuSPARSE Workspace Buffer ---
    size_t bufferSize = 0;
    // Determine the required buffer size for the SpMV operation
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matU, vecX, &beta, vecY,
                                           CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    void *dBuffer = nullptr;
    // Allocate the buffer
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // --- 5. Prepare for Block Iteration ---
    // nBlocks is the number of sub-blocks (size d) the state vector is partitioned into (2^(n-k))
    int nBlocks = 1 << (nQubits - k);

    // Identify qubits NOT in the target list (these define the block index 'blk')
    std::vector<int> nonTargetQubits;
    for (int q = 0; q < nQubits; ++q)
    {
        if (std::find(targetQubits.begin(), targetQubits.end(), q) == targetQubits.end())
        {
            nonTargetQubits.push_back(q);
        }
    }

    // Compute control mask for the qubits that must be '1'
    int controlMask = 0;
    for (int cq : controlQubits)
        controlMask |= (1 << cq);

    // Allocate host and device memory for the state vector offsets
    std::vector<int> h_offsets(d);
    int *d_offsets;
    CHECK_CUDA(cudaMalloc(&d_offsets, d * sizeof(int)));

    // --- 6. Iterate through all computational blocks (Non-Target Qubit States) ---
    for (int blk = 0; blk < nBlocks; ++blk)
    {
        // --- A. Calculate the base offset/mask for the current block ---
        int nonTargetMask = 0;
        // The 'blk' loop index represents the state of the non-target qubits
        for (size_t i = 0; i < nonTargetQubits.size(); ++i)
        {
            if ((blk >> i) & 1)
            {
                nonTargetMask |= (1 << nonTargetQubits[i]);
            }
        }

        // ✅ Skip this block if control qubits not all active
        if ((nonTargetMask & controlMask) != controlMask)
            continue;

        // --- B. Calculate full state vector indices for the current block ---
        // Iterate through all 'd' indices within the current block
        for (int b = 0; b < d; ++b)
        {
            int targetMask = 0;
            // The 'b' index represents the state of the target qubits
            for (int q = 0; q < k; ++q)
            {
                if ((b >> q) & 1)
                {
                    targetMask |= (1 << targetQubits[q]);
                }
            }
            // The full index is the sum of the non-target base mask and the target mask
            h_offsets[b] = nonTargetMask | targetMask;
        }

        // 1. Transfer offsets to device
        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), d * sizeof(int), cudaMemcpyHostToDevice));

        // 2. Gather: Read input sub-vector from global state into scratchpad
        int threads_per_block = 256;
        int blocks = (d + threads_per_block - 1) / threads_per_block;
        gather_kernel CUDA_KERNEL(blocks, threads_per_block)(d_block_in, d_state_in, d_offsets, d);
        CHECK_CUDA(cudaPeekAtLastError());
        cudaDeviceSynchronize();

        // 3. Apply: Perform the Sparse Matrix-Vector Multiplication (SpMV)
        CHECK_CUDA(cudaMemset(d_block_out, 0, d * sizeof(cuDoubleComplex))); // Clear output block
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matU, vecX, &beta, vecY,
                                    CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        CHECK_CUDA(cudaDeviceSynchronize());

        // 4. Scatter: Write output sub-vector from scratchpad back to global state
        scatter_kernel CUDA_KERNEL(blocks, threads_per_block)(d_state_out, d_block_out, d_offsets, d);
        CHECK_CUDA(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }

    // --- 7. Cleanup Resources ---
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_block_in));
    CHECK_CUDA(cudaFree(d_block_out));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroySpMat(matU));

    return 0;
}