#include "SparseDenseConvert.hpp"

#include <vector>
#include <span>
#include <cassert>
#include <cmath>
#include <cuComplex.h>   // cuDoubleComplex, cuCreal, cuCimag

// Return storage lives in the vectors; make spans to pass into libraries.
void dense_to_csr(
    std::span<const cuDoubleComplex> A, // row-major, size n*n
    int n,
    std::vector<int>& row_ptr_out,              // size n+1
    std::vector<int>& col_ind_out,              // size nnz
    std::vector<cuDoubleComplex>& values_out,   // size nnz
    double tol                            // treat |re|,|im| <= tol as zero
) {
    assert(n >= 0);
    assert(static_cast<size_t>(n) * static_cast<size_t>(n) == A.size());

    row_ptr_out.clear();
    col_ind_out.clear();
    values_out.clear();
    row_ptr_out.reserve(static_cast<size_t>(n) + 1);

    auto is_zero = [tol](cuDoubleComplex z) -> bool {
        // zero if both parts within tol
        return std::fabs(cuCreal(z)) <= tol && std::fabs(cuCimag(z)) <= tol;
    };

    row_ptr_out.push_back(0); // row_ptr[0] = 0
    int nnz = 0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const cuDoubleComplex aij = A[static_cast<size_t>(i) * n + j];
            if (!is_zero(aij)) {
                col_ind_out.push_back(j);
                values_out.push_back(aij);
                ++nnz;
            }
        }
        row_ptr_out.push_back(nnz); // cumulative count
    }
}

void csr_to_dense(
    std::span<const int> row_ptr_in,          // size n+1
    std::span<const int> col_ind_in,          // size nnz
    std::span<const cuDoubleComplex> values_in, // size nnz
    int n,
    std::vector<cuDoubleComplex>& A_out       // row-major, size n*n
) {
    assert(n >= 0);
    // Check if row_ptr_in has the correct size (n + 1)
    assert(row_ptr_in.size() == static_cast<size_t>(n) + 1);

    // Get nnz from the last element of row_ptr
    const int nnz = n > 0 ? row_ptr_in[n] : 0;
    
    // Check if col_ind_in and values_in have the correct size (nnz)
    assert(col_ind_in.size() == static_cast<size_t>(nnz));
    assert(values_in.size() == static_cast<size_t>(nnz));

    const size_t dense_size = static_cast<size_t>(n) * static_cast<size_t>(n);

    // Resize the output dense vector and initialize all elements to zero
    A_out.assign(dense_size, zero_cuDoubleComplex());

    // Iterate through each row
    for (int i = 0; i < n; ++i) {
        // Get the starting and ending index in col_ind_in/values_in for the current row i
        const int start_idx = row_ptr_in[i];
        const int end_idx   = row_ptr_in[i + 1];

        // Iterate over the non-zero elements in row i
        for (int k = start_idx; k < end_idx; ++k) {
            
            const int j = col_ind_in[k];            
            const cuDoubleComplex aij = values_in[k];            
            const size_t dense_idx = static_cast<size_t>(i) * n + j;

            assert(j >= 0 && j < n);
            assert(dense_idx < dense_size);

            A_out[dense_idx] = aij;
        }
    }
}