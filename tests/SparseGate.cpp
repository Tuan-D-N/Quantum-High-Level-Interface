#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include "custatevec.h"
#include "cuda_runtime.h"
#include "cusparse.h"
#include "../CuSparseControl/SparseGate.hpp"
#include "../CuQuantumControl/ApplyGates.hpp"

bool deviceArraysNear(const cuDoubleComplex* d1,
                      const cuDoubleComplex* d2,
                      int size,
                      double tol = 1e-12)
{
    // Transfer to host for comparison
    std::vector<cuDoubleComplex> h1(size), h2(size);
    cudaMemcpy(h1.data(), d1, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h2.data(), d2, size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    bool are_near = true;

    for (int i = 0; i < size; i++) {
        double diff_real = std::abs(h1[i].x - h2[i].x);
        double diff_imag = std::abs(h1[i].y - h2[i].y);
        double total_diff = diff_real + diff_imag;

        if (total_diff > tol) {
            // Found a difference, print it and keep checking to find all differences
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
class ApplyGateTest : public ::testing::Test {
protected:
    int nQubits = 4;      // example 4 qubits
    int dim;
    std::vector<cuDoubleComplex> h_sv;
    cuDoubleComplex *d_sv1, *d_sv2;

    void SetUp() override {
        dim = 1 << nQubits;
        h_sv.resize(dim);

        // Initialize random statevector
        for (int i = 0; i < dim; i++) {
            h_sv[i] = make_cuDoubleComplex((double)rand()/RAND_MAX,
                                           (double)rand()/RAND_MAX);
        }

        cudaMalloc(&d_sv1, dim*sizeof(cuDoubleComplex));
        cudaMalloc(&d_sv2, dim*sizeof(cuDoubleComplex));
        cudaMemcpy(d_sv1, h_sv.data(), dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sv2, h_sv.data(), dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cudaFree(d_sv1);
        cudaFree(d_sv2);
    }
};

// Example 2-qubit gate (CNOT)
TEST_F(ApplyGateTest, CompareSparseVsCuStateVec) {
    custatevecHandle_t custateHandle;
    THROW_CUSTATEVECTOR(custatevecCreate(&custateHandle));
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    cusparseHandle_t cusparseHandle;
    THROW_CUSPARSE(cusparseCreate(&cusparseHandle));

    // Sparse 4x4 CNOT CSR (host)
    int h_rowPtr[5] = {0,1,2,4,4};
    int h_colInd[4] = {0,1,3,2};
    cuDoubleComplex h_values[4] = { make_cuDoubleComplex(1,0),
                                    make_cuDoubleComplex(1,0),
                                    make_cuDoubleComplex(1,0),
                                    make_cuDoubleComplex(1,0) };

    // Copy to device
    int *d_rowPtr, *d_colInd;
    cuDoubleComplex *d_values;
    cudaMalloc(&d_rowPtr, 5*sizeof(int));
    cudaMalloc(&d_colInd, 4*sizeof(int));
    cudaMalloc(&d_values, 4*sizeof(cuDoubleComplex));
    cudaMemcpy(d_rowPtr, h_rowPtr, 5*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, h_colInd, 4*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, 4*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Target qubits
    std::vector<int> targets = {1,2}; // example

    // --- Apply custom sparse function
    THROW_BROAD_ERROR(applySparseGate(cusparseHandle, nQubits,
                    d_rowPtr, d_colInd, d_values,
                    d_sv1, d_sv1, targets, 4));


    
    THROW_BROAD_ERROR(applyGatesGeneral<precision::bit_64>(
        custateHandle,
        nQubits,
        h_values,
        false,
        targets,
        {},
        d_sv2,
        extraWorkspace,
        extraWorkspaceSizeInBytes
    ));
    // --- Compare outputs
    EXPECT_TRUE(deviceArraysNear(d_sv1, d_sv2, dim));

    // Cleanup
    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_values);
    cusparseDestroy(cusparseHandle);
}
