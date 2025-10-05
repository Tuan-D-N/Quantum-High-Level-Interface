#include "SparseMatrixExponential.hpp"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <device_launch_parameters.h>
#include "../CudaControl/Helper.hpp"

// ===============================
// Device kernels
// ===============================

// out[i] += (i^k * tmp[i] / k!)
__global__ void add_div_kernel_complex(cuDoubleComplex *out, const cuDoubleComplex *tmp, int n, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    // compute i^k pattern
    cuDoubleComplex i_powk;
    switch (k % 4)
    {
    case 0:
        i_powk = make_cuDoubleComplex(1.0, 0.0);
        break;
    case 1:
        i_powk = make_cuDoubleComplex(0.0, 1.0);
        break;
    case 2:
        i_powk = make_cuDoubleComplex(-1.0, 0.0);
        break;
    case 3:
        i_powk = make_cuDoubleComplex(0.0, -1.0);
        break;
    }

    double invk = 1.0 / static_cast<double>(k);
    cuDoubleComplex scale = make_cuDoubleComplex(invk, 0.0);
    cuDoubleComplex term = cuCmul(i_powk, cuCmul(scale, tmp[i]));
    out[i] = cuCadd(out[i], term);
}

/**
 * @brief Computes the matrix-vector product $v_{out} \approx e^{iA} \cdot v$ using a Taylor series approximation.
 *
 * ===============================
 * Host function: exp(iA)*v ≈ Σ (i^k / k!) A^k v
 * ===============================
 *
 * This host function orchestrates the computation of $e^{iA} \cdot v$ where $A$ is a sparse matrix,
 * using the truncated Taylor series expansion:
 * $$ e^{iA} \cdot v \approx \sum_{k=0}^{\text{order}} \frac{(iA)^k}{k!} \cdot v = \sum_{k=0}^{\text{order}} \frac{i^k}{k!} A^k v $$
 * The calculation is performed on the device (GPU) using cuSPARSE library calls for sparse matrix-vector multiplication (SpMV).
 *
 * The final result is accumulated in the output vector `d_out`.
 *
 * @param handle The cuSPARSE library handle initialized by the caller.
 * @param n The dimension of the square matrix A (number of rows/columns).
 * @param nnz The number of non-zero elements in the sparse matrix A.
 * @param order The maximum order (M) of the Taylor series to compute, i.e., the loop runs from $k=0$ to $k=\text{order}$.
 * @param d_csrRowPtr Pointer to the device array storing the compressed sparse row (CSR) row pointers of matrix A.
 * @param d_csrColInd Pointer to the device array storing the CSR column indices of matrix A.
 * @param d_csrVal Pointer to the device array storing the CSR non-zero values of matrix A (must be a Hermitian matrix, though not explicitly checked here).
 * @param d_out Pointer to the device array holding the input vector $v$ on entry. This vector is overwritten with the final result $\sum_{k=0}^{\text{order}} \frac{i^k}{k!} A^k v$.
 * @return Returns 0 on successful completion.
 */
int expiAv_taylor_cusparse(
    cusparseHandle_t handle,
    int n, int nnz, int order,
    const int *d_csrRowPtr,
    const int *d_csrColInd,
    const cuDoubleComplex *d_csrVal,
    cuDoubleComplex *d_out)
{
    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    // Create sparse matrix descriptor
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, n, n, nnz,
        (void *)d_csrRowPtr, (void *)d_csrColInd, (void *)d_csrVal,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));

    // Allocate single temporary vector
    cuDoubleComplex *d_tmp;
    CHECK_CUDA(cudaMalloc(&d_tmp, n * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMemcpy(d_tmp, d_out, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, d_tmp, CUDA_C_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, n, d_tmp, CUDA_C_64F));

    // Workspace buffer
    size_t bufferSize = 0;
    void *dBuffer = nullptr;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    for (int k = 1; k <= order; ++k)
    {
        // d_tmp = A * d_tmp
        CHECK_CUSPARSE(cusparseSpMV(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY,
            CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        // out += (i^k / k!) * tmp
        add_div_kernel_complex CUDA_KERNEL(blocks, threads)(d_out, d_tmp, n, k);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Cleanup
    cudaFree(dBuffer);
    cudaFree(d_tmp);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matA);
}

// ===============================
// Gather / Scatter kernels
// ===============================
__global__ void gather_kernel(cuDoubleComplex *d_out, const cuDoubleComplex *d_in, const int *d_offsets, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        d_out[i] = d_in[d_offsets[i]];
}

__global__ void scatter_kernel(cuDoubleComplex *d_out, const cuDoubleComplex *d_in, const int *d_offsets, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        d_out[d_offsets[i]] = d_in[i];
}

/**
 * @brief Applies the controlled quantum gate $C(\exp(iA))$ to the state vector using a truncated Taylor series.
 *
 * This function computes the effect of the unitary operator $U = \exp(iA)$ applied to a subset of target qubits,
 * conditionally on the state of a set of control qubits.
 *
 * The operator $U$ on the target qubits is approximated using the Taylor expansion:
 * $$ U \cdot v \approx \sum_{k=0}^{\text{order}} \frac{i^k}{k!} A^k v $$
 *
 * This function handles the "control" logic by selectively applying the matrix powers $A^k$ only to
 * the state vector components where the control qubits are in the correct state (e.g., all $|1\rangle$).
 * The underlying matrix-vector multiplications ($A^k v'$) are performed on the GPU using the cuSPARSE library.
 * The operation is performed *in-place* on the state vector `d_state`.
 *
 * @param handle The cuSPARSE library handle initialized by the caller.
 * @param nQubits The total number of qubits in the quantum system. The size of the state vector is $2^{\text{nQubits}}$.
 * @param d_csrRowPtr Pointer to the device array storing the compressed sparse row (CSR) row pointers of the sparse matrix A (acting on the target qubits space).
 * @param d_csrColInd Pointer to the device array storing the CSR column indices of matrix A.
 * @param d_csrVal Pointer to the device array storing the CSR non-zero values of matrix A.
 * @param d_state Pointer to the device array holding the quantum state vector ($2^{\text{nQubits}}$ complex doubles). The result is written back *in-place*.
 * @param targetQubits A vector containing the indices of the qubits on which the sparse operator $A$ acts. The matrix $A$ must be defined in this subspace.
 * @param controlQubits A vector containing the indices of the qubits that act as controls (typically for the $|1\rangle$ state).
 * @param nnz The number of non-zero elements in the sparse matrix A.
 * @param order The maximum order ($M$) of the Taylor series to compute, which determines the approximation accuracy.
 * @return Returns 0 on successful completion, or a non-zero error code otherwise.
 */
int applyControlledExpTaylor_cusparse(
    cusparseHandle_t handle,
    int nQubits,
    const int *d_csrRowPtr,
    const int *d_csrColInd,
    const cuDoubleComplex *d_csrVal,
    cuDoubleComplex *d_state,
    const std::vector<int> &targetQubits,
    const std::vector<int> &controlQubits,
    int nnz,
    int order)
{
    int k = targetQubits.size();
    int d = 1 << k;
    int dim = 1 << nQubits;
    int nBlocks = 1 << (nQubits - k);

    // Non-target qubits
    std::vector<int> nonTargetQubits;
    for (int q = 0; q < nQubits; ++q)
        if (std::find(targetQubits.begin(), targetQubits.end(), q) == targetQubits.end())
            nonTargetQubits.push_back(q);

    // Control mask
    int controlMask = 0;
    for (int cq : controlQubits)
        controlMask |= (1 << cq);

    // Buffers
    cuDoubleComplex *d_block;
    CHECK_CUDA(cudaMalloc(&d_block, d * sizeof(cuDoubleComplex)));
    std::vector<int> h_offsets(d);
    int *d_offsets;
    CHECK_CUDA(cudaMalloc(&d_offsets, d * sizeof(int)));

    // Iterate over sub-blocks
    for (int blk = 0; blk < nBlocks; ++blk)
    {
        int nonTargetMask = 0;
        for (size_t i = 0; i < nonTargetQubits.size(); ++i)
            if ((blk >> i) & 1)
                nonTargetMask |= (1 << nonTargetQubits[i]);

        if ((nonTargetMask & controlMask) != controlMask)
            continue;

        // Compute sub-block indices
        for (int b = 0; b < d; ++b)
        {
            int targetMask = 0;
            for (int q = 0; q < k; ++q)
                if ((b >> q) & 1)
                    targetMask |= (1 << targetQubits[q]);
            h_offsets[b] = nonTargetMask | targetMask;
        }

        CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), d * sizeof(int), cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (d + threads - 1) / threads;

        // Gather sub-block
        gather_kernel CUDA_KERNEL(blocks, threads)(d_block, d_state, d_offsets, d);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Apply exp(iA) via Taylor
        expiAv_taylor_cusparse(handle, d, nnz, order, d_csrRowPtr, d_csrColInd, d_csrVal, d_block);

        // Scatter back
        scatter_kernel CUDA_KERNEL(blocks, threads)(d_state, d_block, d_offsets, d);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    cudaFree(d_offsets);
    cudaFree(d_block);
    return 0;
}
