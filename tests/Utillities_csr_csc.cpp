#include <gtest/gtest.h>
#include <vector>
#include <cuComplex.h>  // For cuDoubleComplex
#include "../functionality/Utilities.hpp"  // Include the header file with csrToDense and cscToDense



// Test case for csrToDense
TEST(DenseConversionTest, CsrToDense) {
    // Define the CSR format matrix
    std::vector<cuDoubleComplex> values = { cuDoubleComplex{1.0, 0.0}, cuDoubleComplex{2.0, 0.0}, cuDoubleComplex{3.0, 0.0} };
    std::vector<int> rowPtr = { 0, 1, 2, 3 };
    std::vector<int> cols = { 0, 1, 2 };

    int rows = 3;
    int colsCount = 3;

    // Convert CSR to dense
    auto dense = csrToDense(values.data(), rowPtr, cols, rows, colsCount);

    // Expected dense matrix
    std::vector<std::vector<double>> expected_dense = {
        {1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 3.0}
    };

    // Check if the result is correct
    EXPECT_TRUE(matricesEqual(dense, expected_dense));
}

// Test case for cscToDense
TEST(DenseConversionTest, CscToDense) {
    // Define the CSC format matrix
    std::vector<cuDoubleComplex> values = { cuDoubleComplex{1.0, 0.0}, cuDoubleComplex{2.0, 0.0}, cuDoubleComplex{3.0, 0.0} };
    std::vector<int> colPtr = { 0, 1, 2, 3 };
    std::vector<int> rows = { 0, 1, 2 };

    int rowsCount = 3;
    int colsCount = 3;

    // Convert CSC to dense
    auto dense = cscToDense(values.data(), colPtr, rows, rowsCount, colsCount);

    // Expected dense matrix
    std::vector<std::vector<double>> expected_dense = {
        {1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 3.0}
    };

    // Check if the result is correct
    EXPECT_TRUE(matricesEqual(dense, expected_dense));
}

// Additional tests can be added here if needed (e.g., edge cases, larger matrices, etc.)
