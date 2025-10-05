#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <device_launch_parameters.h>

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
    cuDoubleComplex *d_out);

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
    int order);