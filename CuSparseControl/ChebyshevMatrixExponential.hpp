#pragma once

#include "ChebyshevMatrixExponentialCoeff.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <device_launch_parameters.h>
#include "../CudaControl/Helper.hpp" // CHECK_CUDA / CHECK_CUSPARSE
#include "SparseHelper.hpp"          // gather_kernel / scatter_kernel (as in your project)

/**
 * @brief Computes the action of a complex exponential of a sparse matrix on a vector 
 * using the Chebyshev polynomial expansion, with pre-calculated coefficients.
 *
 * This function calculates $v_{out} = (\sum_{k=0}^{\text{order}} \gamma_k A^k) v_{in}$ 
 * where the series is an approximation of $e^{-i t A} v$ (or $e^{+i t A} v$).
 * The coefficients $\gamma_k$ must be pre-calculated, which implicitly includes the time $t$ 
 * and the spectral radius of $A$. This function works directly with device memory.
 *
 * @param handle The cuSPARSE library handle initialized by cusparseCreate().
 * @param n The number of rows/columns of the square sparse matrix A (A is n x n).
 * @param nnz The number of non-zero elements in the sparse matrix A.
 * @param order The maximum polynomial order $m$ used in the Chebyshev expansion (size of gamma - 1).
 * @param d_csrRowPtr Pointer to the device array storing the row pointers of the CSR matrix A.
 * @param d_csrColInd Pointer to the device array storing the column indices of the CSR matrix A.
 * @param d_csrVal Pointer to the device array storing the non-zero values of the CSR matrix A.
 * @param gamma The host vector containing the pre-calculated Chebyshev expansion coefficients $\gamma_k$.
 * @param d_v_in_out Pointer to the device array storing the input vector $v_{in}$. 
 * The result $v_{out}$ is written back to this same location.
 * @return int Returns 0 on success, or a non-zero error code.
 */
int expiAv_chebyshev_gamma_cusparse_device(
    cusparseHandle_t handle,
    int n, int nnz, int order,
    const int *d_csrRowPtr, 
    const int *d_csrColInd, 
    const cuDoubleComplex *d_csrVal,
    const std::vector<cuDoubleComplex> gamma,
    cuDoubleComplex *d_v_in_out);

/**
 * @brief Computes the action of $e^{- i t A}$ on a vector $v$ using the Chebyshev expansion.
 *
 * This function handles the entire process on the host, including calculating the Chebyshev
 * expansion coefficients based on the time step $t$ and applying the polynomial series
 * to the vector using cuSPARSE device routines.
 *
 * @note This function calculates the Chebyshev coefficients internally based on the value of \p t.
 *
 * @param handle The cuSPARSE library handle initialized by cusparseCreate().
 * @param n The number of rows/columns of the square sparse matrix A (A is n x n).
 * @param nnz The number of non-zero elements in the sparse matrix A.
 * @param order The maximum polynomial order $m$ to be used in the Chebyshev expansion.
 * @param h_csrRowPtr Host span containing the row pointers of the CSR matrix A.
 * @param h_csrColInd Host span containing the column indices of the CSR matrix A.
 * @param h_csrVal Host span containing the non-zero values of the CSR matrix A.
 * @param d_v_in_out Pointer to the device array storing the input vector $v_{in}$. 
 * The result $v_{out}$ is written back to this same location.
 * @param t The time factor determining the sign and magnitude of the exponent: 
 * +1 for $e^{-i A}$ (standard evolution), -1 for $e^{+i A}$ (conjugate/reverse evolution).
 * @return int Returns 0 on success, or a non-zero error code.
 */
int expiAv_chebyshev_gamma_cusparse_host(
    cusparseHandle_t handle,
    int n, int nnz, int order,
    std::span<const int> h_csrRowPtr,
    std::span<const int> h_csrColInd,
    std::span<const cuDoubleComplex> h_csrVal,
    cuDoubleComplex *d_v_in_out,
    const double t /* = +1 for exp(-iA), -1 for exp(+iA) */);

/**
 * @brief Applies a controlled exponential of a sparse matrix ($C \otimes e^{-i t A}$) to a quantum state 
 * vector using the Chebyshev expansion with pre-calculated coefficients.
 *
 * This function performs the controlled operation in device memory using the $\gamma_k$ coefficients.
 *
 * @note The $t$ parameter is implicit in the pre-calculated \p gamma coefficients and is therefore not required.
 *
 * @param handle The cuSPARSE library handle initialized by cusparseCreate().
 * @param nQubits The total number of qubits in the system (state vector dimension $2^{\text{nQubits}}$).
 * @param d_csrRowPtr Pointer to the device array storing the row pointers of the target operator $A$.
 * @param d_csrColInd Pointer to the device array storing the column indices of the target operator $A$.
 * @param d_csrVal Pointer to the device array storing the non-zero values of the target operator $A$.
 * @param gamma The host vector containing the pre-calculated Chebyshev expansion coefficients $\gamma_k$.
 * @param d_state Pointer to the device array storing the quantum state vector to be modified.
 * @param targetQubits A host vector listing the target qubits on which the operator $A$ acts.
 * @param controlQubits A host vector listing the control qubits required to be in state $|1\rangle$.
 * @param nnz The number of non-zero elements in the target operator $A$.
 * @param order The maximum polynomial order $m$ used in the Chebyshev expansion.
 * @return int Returns 0 on success, or a non-zero error code.
 */
int applyControlledExpChebyshev_cusparse_device(
    cusparseHandle_t handle,
    int nQubits,
    const int *d_csrRowPtr, 
    const int *d_csrColInd,
    const cuDoubleComplex *d_csrVal,
    const std::vector<cuDoubleComplex> gamma,
    cuDoubleComplex *d_state,
    const std::vector<int> &targetQubits,
    const std::vector<int> &controlQubits,
    int nnz,
    int order);

/**
 * @brief Applies a controlled exponential of a sparse matrix ($C \otimes e^{- i t A}$) to a quantum state 
 * vector, handling the full process from host input to device execution.
 *
 * This function calculates the Chebyshev coefficients based on $t$ and then applies the controlled
 * operation to the state vector in device memory.
 *
 * @param handle The cuSPARSE library handle initialized by cusparseCreate().
 * @param nQubits The total number of qubits in the system (state vector dimension $2^{\text{nQubits}}$).
 * @param d_csrRowPtr Pointer to the device array storing the row pointers of the target operator $A$.
 * @param d_csrColInd Pointer to the device array storing the column indices of the target operator $A$.
 * @param d_csrVal Pointer to the device array storing the non-zero values of the target operator $A$.
 * @param d_state Pointer to the device array storing the quantum state vector to be modified.
 * @param targetQubits A host vector listing the target qubits on which the operator $A$ acts.
 * @param controlQubits A host vector listing the control qubits required to be in state $|1\rangle$.
 * @param nnz The number of non-zero elements in the target operator $A$.
 * @param order The maximum polynomial order $m$ to be used in the Chebyshev expansion.
 * @param t The time factor determining the sign and magnitude of the exponent: 
 * +1 for $e^{-i A}$ (standard evolution), -1 for $e^{+i A}$ (conjugate/reverse evolution).
 * @return int Returns 0 on success, or a non-zero error code.
 */
int applyControlledExpChebyshev_cusparse_host(
    cusparseHandle_t handle,
    int nQubits,
    const int *d_csrRowPtr, 
    const int *d_csrColInd,
    const cuDoubleComplex *d_csrVal,
    cuDoubleComplex *d_state,
    const std::vector<int> &targetQubits,
    const std::vector<int> &controlQubits,
    int nnz,
    int order,
    const double t /* = +1 for exp(-iA), -1 for exp(+iA) */);