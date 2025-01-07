#include <gtest/gtest.h>
#include "../functionality/Utilities.hpp"

TEST(Utilities, isOdd) {
    std::vector<int> oddNumbers{ -21,-9,-7,-5,-3,-1,1,3,5,7,9,11 };
    for (auto number : oddNumbers)
    {
        EXPECT_TRUE(isOdd(number));
    }

    std::vector<int> evenNumbers{ -20,-18,-14,-12,-6,-4,-2,0,2,4,6,8,10,30,50 };
    for (auto number : evenNumbers)
    {
        EXPECT_FALSE(isOdd(number));
    }

}

TEST(Utilities, isEven) {
    std::vector<int> oddNumbers{ -21,-9,-7,-5,-3,-1,1,3,5,7,9,11 };
    for (auto number : oddNumbers)
    {
        EXPECT_FALSE(isEven(number));
    }

    std::vector<int> evenNumbers{ -20,-18,-14,-12,-6,-4,-2,0,2,4,6,8,10,30,50 };
    for (auto number : evenNumbers)
    {
        EXPECT_TRUE(isEven(number));
    }

}


// Test for matricesEqual function
TEST(MatricesEqualTest, EqualMatrices) {
    std::vector<std::vector<double>> matrix1 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    std::vector<std::vector<double>> matrix2 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    EXPECT_TRUE(matricesEqual(matrix1, matrix2));
}

TEST(MatricesEqualTest, EqualMatricesWithTolerance) {
    std::vector<std::vector<double>> matrix1 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    std::vector<std::vector<double>> matrix2 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.000001} // Slight difference within tolerance
    };

    EXPECT_TRUE(matricesEqual(matrix1, matrix2, 1e-5));
}

TEST(MatricesEqualTest, MatricesWithDifferenceOutsideTolerance) {
    std::vector<std::vector<double>> matrix1 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    std::vector<std::vector<double>> matrix2 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 7.0} // Large difference (6.0 vs. 7.0)
    };

    EXPECT_FALSE(matricesEqual(matrix1, matrix2, 1e-5));
}

TEST(MatricesEqualTest, DifferentSizes) {
    std::vector<std::vector<double>> matrix1 = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    std::vector<std::vector<double>> matrix2 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    EXPECT_FALSE(matricesEqual(matrix1, matrix2));
}

TEST(MatricesEqualTest, EmptyMatrices) {
    std::vector<std::vector<double>> matrix1 = {};
    std::vector<std::vector<double>> matrix2 = {};

    EXPECT_TRUE(matricesEqual(matrix1, matrix2));
}

TEST(MatricesEqualTest, EmptyMatrixAndNonEmptyMatrix) {
    std::vector<std::vector<double>> matrix1 = {};
    std::vector<std::vector<double>> matrix2 = {{1.0, 2.0, 3.0}};

    EXPECT_FALSE(matricesEqual(matrix1, matrix2));
}

TEST(MatricesEqualTest, DifferentMatrices) {
    std::vector<std::vector<double>> matrix1 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    std::vector<std::vector<double>> matrix2 = {
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 12.0}
    };

    EXPECT_FALSE(matricesEqual(matrix1, matrix2));
}