#pragma once
#include <vector>
#include <span>
#include <cassert>
#include <cmath>
#include <cuComplex.h>   // cuDoubleComplex, cuCreal, cuCimag

constexpr inline cuDoubleComplex zero_cuDoubleComplex() { return {0.0, 0.0}; }


// Return storage lives in the vectors; make spans to pass into libraries.
void dense_to_csr(
    std::span<const cuDoubleComplex> A, // row-major, size n*n
    int n,
    std::vector<int>& row_ptr_out,              // size n+1
    std::vector<int>& col_ind_out,              // size nnz
    std::vector<cuDoubleComplex>& values_out,   // size nnz
    double tol = 0.0                             // treat |re|,|im| <= tol as zero
);

void csr_to_dense(
    std::span<const int> row_ptr_in,          // size n+1
    std::span<const int> col_ind_in,          // size nnz
    std::span<const cuDoubleComplex> values_in, // size nnz
    int n,
    std::vector<cuDoubleComplex>& A_out       // row-major, size n*n
);