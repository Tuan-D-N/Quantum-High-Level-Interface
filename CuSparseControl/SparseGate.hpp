#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <vector>
#include <cusparse.h>
#include <stdlib.h>           // EXIT_FAILURE
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.

#ifdef __CUDACC__
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#else
#define CUDA_KERNEL(...)
#endif

/// @brief Applies a sparse quantum gate U (in CSR format) to a quantum state vector.
///        The operation performed is: d_state_out = U * d_state_in.
/// @param handle The cuSPARSE library handle.
/// @param nQubits The total number of qubits in the system (defines vector and matrix dimension $N=2^{\text{nQubits}}$).
/// @param d_csrRowPtrU Device pointer to the CSR row pointers of the sparse gate U.
/// @param d_csrColIndU Device pointer to the CSR column indices of the sparse gate U.
/// @param d_csrValU Device pointer to the non-zero values of the sparse gate U (cuDoubleComplex for quantum state).
/// @param d_state_in Device pointer to the input quantum state vector (dense, $|\psi_{\text{in}}\rangle$).
/// @param d_state_out Device pointer to the output quantum state vector (dense, $|\psi_{\text{out}}\rangle$).
/// @param targetQubits List of target qubits that the gate U operates on (passed as context, not directly used in the raw SpMV call).
/// @param nnzU The number of non-zero elements in the sparse gate U.
/// @return An integer status code (0 for success, non-zero for error).
int applySparseGate(
    cusparseHandle_t handle,
    int nQubits,
    const int *d_csrRowPtrU,
    const int *d_csrColIndU,
    const cuDoubleComplex *d_csrValU,
    const cuDoubleComplex *d_state_in,
    cuDoubleComplex *d_state_out,
    const std::vector<int> &targetQubits,
    int nnzU);

/// @brief Applies a multi-controlled sparse quantum gate CU (in CSR format) to a quantum state vector.
///        The operation performed is: d_state_out = CU * d_state_in.
///        Note: The CU matrix itself must already be constructed to implement the multi-control logic.
/// @param handle The cuSPARSE library handle.
/// @param nQubits The total number of qubits in the system (defines vector and matrix dimension $N=2^{\text{nQubits}}$).
/// @param d_csrRowPtrU Device pointer to the CSR row pointers of the sparse controlled gate CU.
/// @param d_csrColIndU Device pointer to the CSR column indices of the sparse controlled gate CU.
/// @param d_csrValU Device pointer to the non-zero values of the sparse controlled gate CU (cuDoubleComplex for quantum state).
/// @param d_state_in Device pointer to the input quantum state vector (dense, $|\psi_{\text{in}}\rangle$).
/// @param d_state_out Device pointer to the output quantum state vector (dense, $|\psi_{\text{out}}\rangle$).
/// @param controlQubits Vector of indices for the control qubits. The gate U is applied only if all these qubits are in state $|1\rangle$.
/// @param targetQubits List of target qubits that the gate U operates on (passed as context, not directly used in the raw SpMV call).
/// @param nnzU The number of non-zero elements in the sparse gate CU.
/// @return An integer status code (0 for success, non-zero for error).
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
    int nnzU);