#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <tuple>
#include <numeric>
#include <random>
#include <algorithm>
#include <complex>

#include "custatevec.h"
#include "cuda_runtime.h"
#include "cusparse.h"
#include "../CuSparseControl/SparseGate.hpp"
#include "../CuQuantumControl/ApplyGates.hpp"

// ... (existing helper functions and test fixture) ...
bool deviceArraysNear(const cuDoubleComplex *d1,
                      const cuDoubleComplex *d2,
                      int size,
                      double tol = 1e-12)
{
    // Transfer to host for comparison
    std::vector<cuDoubleComplex> h1(size), h2(size);
    cudaMemcpy(h1.data(), d1, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h2.data(), d2, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    bool are_near = true;

    for (int i = 0; i < size; i++)
    {
        double diff_real = std::abs(h1[i].x - h2[i].x);
        double diff_imag = std::abs(h1[i].y - h2[i].y);
        double total_diff = diff_real + diff_imag;

        if (total_diff > tol)
        {
            std::cerr << "Difference at index " << i << ":\n"
                      << "  d_sv1[" << i << "] = " << h1[i].x << " + " << h1[i].y << "i\n"
                      << "  d_sv2[" << i << "] = " << h2[i].x << " + " << h2[i].y << "i\n"
                      << "  Total difference = " << total_diff << " (tolerance = " << tol << ")\n";
            are_near = false;
        }
    }
    return are_near;
}

// Test fixture
class ApplyGateSparse : public ::testing::TestWithParam<std::tuple<int, int>>
{
protected:
    int nQubits;
    int kQubits;
    int dim;
    std::vector<cuDoubleComplex> h_sv;
    cuDoubleComplex *d_sv1, *d_sv2;

    void SetUp() override
    {
        std::tie(nQubits, kQubits) = GetParam();
        dim = 1 << nQubits;
        h_sv.resize(dim);

        // Initialize random statevector
        for (int i = 0; i < dim; i++)
        {
            h_sv[i] = make_cuDoubleComplex(static_cast<double>(i), static_cast<double>(i));
        }

        cudaMalloc(&d_sv1, dim * sizeof(cuDoubleComplex));
        cudaMalloc(&d_sv2, dim * sizeof(cuDoubleComplex));
        cudaMemcpy(d_sv1, h_sv.data(), dim * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sv2, h_sv.data(), dim * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    }

    void TearDown() override
    {
        cudaFree(d_sv1);
        cudaFree(d_sv2);
    }
};

TEST_P(ApplyGateSparse, CompareSparseVsCuStateVec_PauliX)
{
    if (kQubits != 1)
    {
        GTEST_SKIP() << "Skipping Pauli-X test for kQubits != 1";
    }

    for (int target_n = 0; target_n < nQubits; ++target_n)
    {
        custatevecHandle_t custateHandle;
        THROW_CUSTATEVECTOR(custatevecCreate(&custateHandle));
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        cusparseHandle_t cusparseHandle;
        THROW_CUSPARSE(cusparseCreate(&cusparseHandle));

        // Sparse 2x2 Pauli-X CSR (host)
        int h_rowPtr_X[3] = {0, 1, 2};
        int h_colInd_X[2] = {1, 0};
        cuDoubleComplex h_values_sparse_X[2] = {
            make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0)};

        int *d_rowPtr_X, *d_colInd_X;
        cuDoubleComplex *d_values_sparse_X;
        cudaMalloc(&d_rowPtr_X, 3 * sizeof(int));
        cudaMalloc(&d_colInd_X, 2 * sizeof(int));
        cudaMalloc(&d_values_sparse_X, 2 * sizeof(cuDoubleComplex));
        cudaMemcpy(d_rowPtr_X, h_rowPtr_X, 3 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colInd_X, h_colInd_X, 2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values_sparse_X, h_values_sparse_X, 2 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        // Create dense 2x2 Pauli-X matrix for cuQuantum (column-major)
        cuDoubleComplex h_dense_matrix_X[4] = {
            make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0),
            make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0)};

        cuDoubleComplex *d_dense_matrix_X;
        cudaMalloc(&d_dense_matrix_X, 4 * sizeof(cuDoubleComplex));
        cudaMemcpy(d_dense_matrix_X, h_dense_matrix_X, 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        std::vector<int> targets = {target_n};

        THROW_BROAD_ERROR(applySparseGate(cusparseHandle, nQubits, d_rowPtr_X, d_colInd_X, d_values_sparse_X,
                                          d_sv1, d_sv1, targets, 2));

        THROW_BROAD_ERROR(applyGatesGeneral<precision::bit_64>(
            custateHandle,
            nQubits,
            std::span(h_dense_matrix_X),
            false,
            targets,
            {},
            d_sv2,
            extraWorkspace,
            extraWorkspaceSizeInBytes));

        EXPECT_TRUE(deviceArraysNear(d_sv1, d_sv2, dim));

        cudaFree(d_rowPtr_X);
        cudaFree(d_colInd_X);
        cudaFree(d_values_sparse_X);
        cudaFree(d_dense_matrix_X);
        cusparseDestroy(cusparseHandle);
        custatevecDestroy(custateHandle);
    }
}

// ... (existing test cases like CNOT and Hadamard) ...
TEST_P(ApplyGateSparse, CompareSparseVsCuStateVec_CNOT)
{
    if (kQubits != 2)
    {
        GTEST_SKIP() << "Skipping CNOT test for kQubits != 2";
    }

    // Since kQubits is a parameter, you need to select two target qubits
    // dynamically. We can iterate through all valid pairs.
    for (int control_n = 0; control_n < nQubits; ++control_n)
    {
        for (int target_n = 0; target_n < nQubits; ++target_n)
        {
            if (control_n == target_n)
                continue;

            custatevecHandle_t custateHandle;
            THROW_CUSTATEVECTOR(custatevecCreate(&custateHandle));
            void *extraWorkspace = nullptr;
            size_t extraWorkspaceSizeInBytes = 0;

            cusparseHandle_t cusparseHandle;
            THROW_CUSPARSE(cusparseCreate(&cusparseHandle));

            // Sparse 4x4 CNOT CSR (host)
            int h_rowPtr[5] = {0, 1, 2, 3, 4};
            int h_colInd[4] = {0, 1, 3, 2};
            cuDoubleComplex h_values_sparse[4] = {
                make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0),
                make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0)};

            int *d_rowPtr, *d_colInd;
            cuDoubleComplex *d_values_sparse;
            cudaMalloc(&d_rowPtr, 5 * sizeof(int));
            cudaMalloc(&d_colInd, 4 * sizeof(int));
            cudaMalloc(&d_values_sparse, 4 * sizeof(cuDoubleComplex));
            cudaMemcpy(d_rowPtr, h_rowPtr, 5 * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_colInd, h_colInd, 4 * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_values_sparse, h_values_sparse, 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

            cuDoubleComplex h_dense_matrix_transposed[16] = {
                make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
                make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
                make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0),
                make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0)};

            cuDoubleComplex *d_dense_matrix;
            cudaMalloc(&d_dense_matrix, 16 * sizeof(cuDoubleComplex));
            cudaMemcpy(d_dense_matrix, h_dense_matrix_transposed, 16 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

            std::vector<int> target_controls = {control_n, target_n}; // Example: Qubit 1 is control, Qubit 2 is target

            THROW_BROAD_ERROR(applySparseGate(cusparseHandle, nQubits, d_rowPtr, d_colInd, d_values_sparse,
                                              d_sv1, d_sv1, target_controls, 4));

            THROW_BROAD_ERROR(applyGatesGeneral<precision::bit_64>(
                custateHandle,
                nQubits,
                std::span(h_dense_matrix_transposed),
                false,
                target_controls,
                {},
                d_sv2,
                extraWorkspace,
                extraWorkspaceSizeInBytes));

            EXPECT_TRUE(deviceArraysNear(d_sv1, d_sv2, dim));

            cudaFree(d_rowPtr);
            cudaFree(d_colInd);
            cudaFree(d_values_sparse);
            cudaFree(d_dense_matrix);
            cusparseDestroy(cusparseHandle);
            custatevecDestroy(custateHandle);
        }
    }
}

TEST_P(ApplyGateSparse, CompareSparseVsCuStateVec_DeterministicGate)
{
    // Skip if kQubits is too large or too small
    if (kQubits < 1 || kQubits > 10)
    {
        GTEST_SKIP() << "Skipping deterministic gate test for kQubits = " << kQubits;
    }

    custatevecHandle_t custateHandle;
    THROW_CUSTATEVECTOR(custatevecCreate(&custateHandle));
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    cusparseHandle_t cusparseHandle;
    THROW_CUSPARSE(cusparseCreate(&cusparseHandle));

    // Determine gate size
    int d = 1 << kQubits;
    int dense_size = d * d;

    // --- Generate a deterministic gate matrix
    std::vector<cuDoubleComplex> h_dense_matrix(dense_size);
    for (int i = 0; i < d; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            h_dense_matrix[i * d + j] = make_cuDoubleComplex(
                static_cast<double>(i + j) / static_cast<double>(d),
                static_cast<double>(i - j) / static_cast<double>(d));
        }
    }

    // --- Convert the dense matrix to CSR format
    std::vector<int> h_rowPtr_sparse;
    std::vector<int> h_colInd_sparse;
    std::vector<cuDoubleComplex> h_values_sparse;

    h_rowPtr_sparse.push_back(0);
    for (int row = 0; row < d; ++row)
    {
        for (int col = 0; col < d; ++col)
        {
            cuDoubleComplex val = h_dense_matrix[row * d + col];
            // Treat very small values as zero for sparsity
            if (std::abs(val.x) > 1e-15 || std::abs(val.y) > 1e-15)
            {
                h_colInd_sparse.push_back(col);
                h_values_sparse.push_back(val);
            }
        }
        h_rowPtr_sparse.push_back(h_values_sparse.size());
    }

    int nnz_sparse = h_values_sparse.size();

    // --- Copy sparse data to device
    int *d_rowPtr_sparse, *d_colInd_sparse;
    cuDoubleComplex *d_values_sparse;
    cudaMalloc(&d_rowPtr_sparse, (d + 1) * sizeof(int));
    cudaMalloc(&d_colInd_sparse, nnz_sparse * sizeof(int));
    cudaMalloc(&d_values_sparse, nnz_sparse * sizeof(cuDoubleComplex));
    cudaMemcpy(d_rowPtr_sparse, h_rowPtr_sparse.data(), (d + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd_sparse, h_colInd_sparse.data(), nnz_sparse * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_sparse, h_values_sparse.data(), nnz_sparse * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // --- Copy dense matrix to device
    cuDoubleComplex *d_dense_matrix;
    cudaMalloc(&d_dense_matrix, dense_size * sizeof(cuDoubleComplex));
    cudaMemcpy(d_dense_matrix, h_dense_matrix.data(), dense_size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Select k random target qubits
    std::vector<int> targetQubits;
    std::vector<int> allQubits(nQubits);
    std::iota(allQubits.begin(), allQubits.end(), 0);

    // Use a fixed seed for reproducibility
    std::mt19937 gen(0);
    std::shuffle(allQubits.begin(), allQubits.end(), gen);

    for (int i = 0; i < kQubits; ++i)
    {
        targetQubits.push_back(allQubits[i]);
    }
    // std::sort(targetQubits.begin(), targetQubits.end());

    // --- Apply custom sparse function
    THROW_BROAD_ERROR(applySparseGate(cusparseHandle, nQubits, d_rowPtr_sparse, d_colInd_sparse, d_values_sparse,
                                      d_sv1, d_sv1, targetQubits, nnz_sparse));

    // --- Apply cuQuantum's applyGatesGeneral with the dense matrix
    THROW_BROAD_ERROR(applyGatesGeneral<precision::bit_64>(
        custateHandle,
        nQubits,
        std::span(h_dense_matrix),
        true, // Matrix is on host
        targetQubits,
        {},
        d_sv2,
        extraWorkspace,
        extraWorkspaceSizeInBytes));

    // --- Compare outputs
    EXPECT_TRUE(deviceArraysNear(d_sv1, d_sv2, dim));

    // Cleanup
    cudaFree(d_rowPtr_sparse);
    cudaFree(d_colInd_sparse);
    cudaFree(d_values_sparse);
    cudaFree(d_dense_matrix);
    cusparseDestroy(cusparseHandle);
    custatevecDestroy(custateHandle);
}

// Global instantiation of test suite
INSTANTIATE_TEST_SUITE_P(
    SparseGateTestSuite,
    ApplyGateSparse,
    ::testing::ValuesIn([]
                        {
        std::vector<std::tuple<int, int>> params;
        for (int n = 2; n <= 8; ++n) {
            for (int k = 1; k <= 3; ++k) {
                if (k <= n) {
                    params.emplace_back(n, k);
                }
            }
        }
        return params; }()));