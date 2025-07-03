#include <gtest/gtest.h>
#include "../CuSparseControl/ApplySparseCSRMat.hpp"



cuDoubleComplex c(double r, double i) {
    return make_cuDoubleComplex(r, i);
}

TEST(ApplySparseCSRMat, SparseMatAppliesCorrectly) {
    cusparseHandle_t handle;
    ASSERT_EQ(cusparseCreate(&handle), CUSPARSE_STATUS_SUCCESS);

    // Matrix: [1+i, 0; 0, 0]
    std::vector<int> csrOffsets = {0, 1};       // 2 rows
    std::vector<int> csrCols = {0};                // one non-zero at row 0, col 0
    std::vector<cuDoubleComplex> values = {c(1.0, 1.0)};

    // Input statevector: [1, 2]
    std::vector<cuDoubleComplex> h_input = {c(1.0, 0.0), c(2.0, 0.0)};
    std::vector<cuDoubleComplex> h_output(2);
    std::vector<cuDoubleComplex> h_expected = {c(1.0, 1.0), c(0.0, 0.0)}; // Expected result

    // Allocate and copy to device
    cuDoubleComplex *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(cuDoubleComplex) * 2);
    cudaMalloc(&d_output, sizeof(cuDoubleComplex) * 2);
    cudaMemcpy(d_input, h_input.data(), sizeof(cuDoubleComplex) * 2, cudaMemcpyHostToDevice);

    std::span<cuDoubleComplex> svIn(d_input, 2);
    std::span<cuDoubleComplex> svOut(d_output, 2);

    // Run the function
    EXPECT_EQ(applySparseCSRMat(handle, csrOffsets, csrCols, values, svIn, svOut), cudaSuccess);

    // Copy back result
    cudaMemcpy(h_output.data(), d_output, sizeof(cuDoubleComplex) * 2, cudaMemcpyDeviceToHost);

    // Compare each element
    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(h_output[i].x, h_expected[i].x, 1e-6);
        EXPECT_NEAR(h_output[i].y, h_expected[i].y, 1e-6);
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cusparseDestroy(handle);
}

TEST(ApplySparseCSRMat, Full2x2UnitaryMatrix) {
    cusparseHandle_t handle;
    ASSERT_EQ(cusparseCreate(&handle), CUSPARSE_STATUS_SUCCESS);

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    // Matrix:
    // [ 1/√2,  1/√2 ]
    // [ 1/√2, -1/√2 ]
    std::vector<int> csrOffsets = {0, 2, 4};
    std::vector<int> csrCols = {0, 1, 0, 1};
    std::vector<cuDoubleComplex> values = {
        c(inv_sqrt2, 0.0), c(inv_sqrt2, 0.0),
        c(inv_sqrt2, 0.0), c(-inv_sqrt2, 0.0)
    };

    // Input vector: [1, 0]
    std::vector<cuDoubleComplex> h_input = {c(1.0, 0.0), c(0.0, 0.0)};
    std::vector<cuDoubleComplex> h_output(2);
    std::vector<cuDoubleComplex> h_expected = {
        c(inv_sqrt2, 0.0),  // (1/√2)*1 + (1/√2)*0
        c(inv_sqrt2, 0.0)   // (1/√2)*1 + (-1/√2)*0
    };

    cuDoubleComplex *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(cuDoubleComplex) * 2);
    cudaMalloc(&d_output, sizeof(cuDoubleComplex) * 2);
    cudaMemcpy(d_input, h_input.data(), sizeof(cuDoubleComplex) * 2, cudaMemcpyHostToDevice);

    std::span<cuDoubleComplex> svIn(d_input, 2);
    std::span<cuDoubleComplex> svOut(d_output, 2);

    EXPECT_EQ(applySparseCSRMat(handle, csrOffsets, csrCols, values, svIn, svOut), cudaSuccess);
    cudaMemcpy(h_output.data(), d_output, sizeof(cuDoubleComplex) * 2, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(h_output[i].x, h_expected[i].x, 1e-6);
        EXPECT_NEAR(h_output[i].y, h_expected[i].y, 1e-6);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cusparseDestroy(handle);
}

TEST(ApplySparseCSRMat, FourByFourSparseGate) {
    cusparseHandle_t handle;
    ASSERT_EQ(cusparseCreate(&handle), CUSPARSE_STATUS_SUCCESS);

    // Matrix: 4x4
    std::vector<int> csrOffsets = {0, 1, 2, 3, 4};  // row starts
    std::vector<int> csrCols = {0, 2, 1, 3};        // col indices
    std::vector<cuDoubleComplex> values = {
        c(1.0, 0.0),    // row 0: [1, 0, 0, 0]
        c(1.0, 0.0),    // row 1: [0, 0, 1, 0]
        c(1.0, 0.0),    // row 2: [0, 1, 0, 0]
        c(-1.0, 0.0)    // row 3: [0, 0, 0, -1]
    };

    // Input vector: [1, 0, 0, 0]
    std::vector<cuDoubleComplex> h_input = {
        c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)
    };

    std::vector<cuDoubleComplex> h_output(4);
    std::vector<cuDoubleComplex> h_expected = {
        c(1.0, 0.0),  // row 0 * input = 1
        c(0.0, 0.0),  // row 1 * input = 0
        c(0.0, 0.0),  // row 2 * input = 0
        c(0.0, 0.0)   // row 3 * input = 0
    };

    // Device alloc + copy
    cuDoubleComplex *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(cuDoubleComplex) * 4);
    cudaMalloc(&d_output, sizeof(cuDoubleComplex) * 4);
    cudaMemcpy(d_input, h_input.data(), sizeof(cuDoubleComplex) * 4, cudaMemcpyHostToDevice);

    std::span<cuDoubleComplex> svIn(d_input, 4);
    std::span<cuDoubleComplex> svOut(d_output, 4);

    EXPECT_EQ(applySparseCSRMat(handle, csrOffsets, csrCols, values, svIn, svOut), cudaSuccess);
    cudaMemcpy(h_output.data(), d_output, sizeof(cuDoubleComplex) * 4, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(h_output[i].x, h_expected[i].x, 1e-6);
        EXPECT_NEAR(h_output[i].y, h_expected[i].y, 1e-6);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cusparseDestroy(handle);
}
