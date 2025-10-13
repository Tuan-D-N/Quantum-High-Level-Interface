#pragma once
#include <cuComplex.h>   // cuDoubleComplex, make_cuDoubleComplex, cuCadd, cuCmul, ...
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <optional>
#include <span>

/**
 * @brief Computes the exponential of a sparse matrix $A$, $e^{tA}$, using a 
 * Chebyshev polynomial spectral method, and returns the result (typically 
 * $e^{tA}v$ or a spectral representation of $e^{tA}$).
 *
 * This function calculates the action of the matrix exponential $e^{-itA}$ 
 * using a truncated Chebyshev polynomial expansion of degree $m$. The matrix $A$
 * is provided in the Compressed Sparse Row (CSR) format.
 * The method requires $A$ to be scaled to map its spectrum onto the interval $[-1, 1]$,
 * using the spectral radius $\rho(A)$.
 *
 * @param n The dimension of the square matrix $A$ (i.e., $A$ is $n \times n$).
 * @param row_ptr A span of integers representing the CSR row pointer array. 
 * Length is $n + 1$.
 * @param col_ind A span of integers representing the CSR column index array. 
 * Length is $nnz$ (number of non-zero elements).
 * @param values A span of complex double-precision numbers 
 * (using cuDoubleComplex, typically a struct with double real and imag parts)
 * representing the non-zero values of the matrix $A$. Length is $nnz$.
 * @param t The time or scaling parameter in the matrix exponential $exp(-i t A)$.
 * @param m The degree of the Chebyshev polynomial expansion to use for the approximation.
 * @param spectral_radius An optional double-precision value specifying the spectral radius $\rho(A)$ of the matrix $A$.
 * If provided, this value is used for the scaling and shifting of $A$.
 * If std::nullopt (or omitted), the function will estimate or compute $\rho(A)$.
 * @return std::vector<cuDoubleComplex> The result of the computation. 
 * Depending on the implementation, this could be:
 * 1. The resulting vector $exp(-i t A)v$ (if a vector $v$ is implicitly or explicitly part of the state).
 * 2. The computed Chebyshev expansion coefficients.
 * 3. The full matrix $exp(-i t A)$ (unlikely for a sparse method).
 * The return type contains $n$ elements.
 */
std::vector<cuDoubleComplex>
chebyshev_exp_gamma_spectral_csr(
    int n,
    std::span<const int> row_ptr,
    std::span<const int> col_ind,
    std::span<const cuDoubleComplex> values,
    double t,
    int m,
    std::optional<double> spectral_radius = std::nullopt);