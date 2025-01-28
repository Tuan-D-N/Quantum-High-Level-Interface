#include <gtest/gtest.h>
#include <vector>
#include <span>
#include <algorithm>
#include <iostream>
#include <complex>
#include "../CudaQControl/stateContracter.hpp"
#include "../functionality/Utilities.hpp"

// Test fixture for GetStateProbability
class GetStateProbabilityTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Set up test data
        d_sv_float = {{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {0.0f, 0.0f}};
        d_sv_double = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {0.0, 0.0}};
        qubitsToCare = {0, 1};
        out_AssociatedProbability = {};
    }

    // Test data
    std::vector<cuFloatComplex> d_sv_float;
    std::vector<cuDoubleComplex> d_sv_double;
    std::vector<int> qubitsToCare;
    std::vector<double> out_AssociatedProbability;
};

// Test for float precision
TEST_F(GetStateProbabilityTest, FloatPrecision)
{
    int nQubits = 2;
    int result = GetStateProbability<precision::bit_32>(nQubits, d_sv_float.data(), qubitsToCare, out_AssociatedProbability);
    EXPECT_EQ(result, 0); // Check if the function returns success
}

// Test for double precision
TEST_F(GetStateProbabilityTest, DoublePrecision)
{
    int nQubits = 2;
    int result = GetStateProbability<precision::bit_64>(nQubits, d_sv_double.data(), qubitsToCare, out_AssociatedProbability);
    EXPECT_EQ(result, 0); // Check if the function returns success
}

// Test with empty qubitsToCare
TEST_F(GetStateProbabilityTest, EmptyQubitsToCare)
{
    int nQubits = 2;
    std::vector<int> emptyQubitsToCare = {};
    int result = GetStateProbability<precision::bit_64>(nQubits, d_sv_double.data(), emptyQubitsToCare, out_AssociatedProbability);
    EXPECT_EQ(result, 0); // Check if the function returns success
}

// Test with invalid qubitsToCare
TEST_F(GetStateProbabilityTest, InvalidQubitsToCare)
{
    int nQubits = 2;
    std::vector<int> invalidQubitsToCare = {3, 4}; // Qubits out of range
    int result = GetStateProbability<precision::bit_64>(nQubits, d_sv_double.data(), invalidQubitsToCare, out_AssociatedProbability);
    EXPECT_EQ(result, 0); // Check if the function returns success
}