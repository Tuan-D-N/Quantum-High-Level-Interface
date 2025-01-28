#include <gtest/gtest.h>
#include <complex>
#include <span>
#include <array>
#include <utility>
#include <stdexcept>
#include "../CudaQControl/stateContracter.hpp"


// Test fixture for the measure1QubitUnified function
class Measure1QubitUnifiedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up common test data
        validState = { {1.0, 0.0}, {0.0, 0.0} }; // |0⟩ state
        normalizedState = { {1.0 / std::sqrt(2), 0.0}, {1.0 / std::sqrt(2), 0.0} }; // (|0⟩ + |1⟩)/√2
        unnormalizedState = { {2.0, 0.0}, {2.0, 0.0} }; // Unnormalized state
        oddSizedState = { {1.0, 0.0}, {0.0, 0.0}, {1.0, 0.0} }; // Invalid size
    }

    // Test data
    std::vector<cuDoubleComplex> validState;
    std::vector<cuDoubleComplex> normalizedState;
    std::vector<cuDoubleComplex> unnormalizedState;
    std::vector<cuDoubleComplex> oddSizedState;
};

// Test valid input: |0⟩ state
TEST_F(Measure1QubitUnifiedTest, ValidInputZeroState) {
    auto result = measure1QubitUnified<precision::bit_64>(validState);
    EXPECT_DOUBLE_EQ(result.first, 1.0);  // Probability of |0⟩
    EXPECT_DOUBLE_EQ(result.second, 0.0); // Probability of |1⟩
}

// Test valid input: (|0⟩ + |1⟩)/√2 state
TEST_F(Measure1QubitUnifiedTest, ValidInputSuperpositionState) {
    auto result = measure1QubitUnified<precision::bit_64>(normalizedState);
    EXPECT_DOUBLE_EQ(result.first, 0.5);  // Probability of |0⟩
    EXPECT_DOUBLE_EQ(result.second, 0.5); // Probability of |1⟩
}

// Test unnormalized input
TEST_F(Measure1QubitUnifiedTest, UnnormalizedInput) {
    auto result = measure1QubitUnified<precision::bit_64>(unnormalizedState);
    EXPECT_DOUBLE_EQ(result.first, 0.5);  // Probability of |0⟩ (normalized)
    EXPECT_DOUBLE_EQ(result.second, 0.5); // Probability of |1⟩ (normalized)
}

// Test invalid input: odd-sized state
TEST_F(Measure1QubitUnifiedTest, InvalidInputOddSize) {
    EXPECT_THROW(measure1QubitUnified<precision::bit_64>(oddSizedState), std::invalid_argument);
}

// Test edge case: empty input
TEST_F(Measure1QubitUnifiedTest, EmptyInput) {
    std::vector<cuDoubleComplex> emptyState;
    EXPECT_THROW(measure1QubitUnified<precision::bit_64>(emptyState), std::invalid_argument);
}

// Test edge case: all zeros
TEST_F(Measure1QubitUnifiedTest, AllZerosInput) {
    std::vector<cuDoubleComplex> zeroState = { {0.0, 0.0}, {0.0, 0.0} };
    auto result = measure1QubitUnified<precision::bit_64>(zeroState);
    EXPECT_TRUE(std::isnan(result.first));  // Probability of |0⟩ is NaN (division by zero)
    EXPECT_TRUE(std::isnan(result.second)); // Probability of |1⟩ is NaN (division by zero)
}