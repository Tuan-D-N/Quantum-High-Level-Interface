#include <gtest/gtest.h>
#include <cuComplex.h>
#include <vector>
#include "../functionality/Transpose.hpp"

// Helper to compare cuDoubleComplex values
bool compareCuDoubleComplex(const cuDoubleComplex &a, const cuDoubleComplex &b)
{
    return cuCreal(a) == cuCreal(b) && cuCimag(a) == cuCimag(b);
}

// Helper to compare arrays of cuDoubleComplex
bool compareCuDoubleComplexArrays(const cuDoubleComplex *a, const cuDoubleComplex *b, int size)
{
    for (int i = 0; i < size; ++i)
    {
        if (!compareCuDoubleComplex(a[i], b[i]))
            return false;
    }
    return true;
}

TEST(TransposeTest, SquareMatrix)
{
    std::vector<std::vector<int>> matrix = {{1, 2, 3},
                                            {4, 5, 6},
                                            {7, 8, 9}};
    std::vector<std::vector<int>> expected = {{1, 4, 7},
                                              {2, 5, 8},
                                              {3, 6, 9}};
    EXPECT_EQ(Transpose(matrix), expected);
}

TEST(TransposeTest, RectangularMatrix)
{
    std::vector<std::vector<int>> matrix = {{1, 2, 3},
                                            {4, 5, 6}};
    std::vector<std::vector<int>> expected = {{1, 4},
                                              {2, 5},
                                              {3, 6}};
    EXPECT_EQ(Transpose(matrix), expected);
}

TEST(TransposeTest, SingleRowMatrix)
{
    std::vector<std::vector<int>> matrix = {{1, 2, 3}};
    std::vector<std::vector<int>> expected = {{1},
                                              {2},
                                              {3}};
    EXPECT_EQ(Transpose(matrix), expected);
}

TEST(TransposeTest, SingleColumnMatrix)
{
    std::vector<std::vector<int>> matrix = {{1},
                                            {2},
                                            {3}};
    std::vector<std::vector<int>> expected = {{1, 2, 3}};
    EXPECT_EQ(Transpose(matrix), expected);
}

TEST(TransposeTest, EmptyMatrix)
{
    std::vector<std::vector<int>> matrix = {};
    std::vector<std::vector<int>> expected = {};
    EXPECT_EQ(Transpose(matrix), expected);
}

// Test case for a square matrix
TEST(TransposeCuDoubleComplexTest, SquareMatrix)
{
    int rows = 2, cols = 2;
    cuDoubleComplex input[] = {
        make_cuDoubleComplex(1.0, 0.1), make_cuDoubleComplex(2.0, 0.2),
        make_cuDoubleComplex(3.0, 0.3), make_cuDoubleComplex(4.0, 0.4)};
    cuDoubleComplex expected[] = {
        make_cuDoubleComplex(1.0, 0.1), make_cuDoubleComplex(3.0, 0.3),
        make_cuDoubleComplex(2.0, 0.2), make_cuDoubleComplex(4.0, 0.4)};
    cuDoubleComplex output[4];

    Transpose(input, output, rows, cols);

    ASSERT_TRUE(compareCuDoubleComplexArrays(output, expected, rows * cols));
}

// Test case for a rectangular matrix
TEST(TransposeCuDoubleComplexTest, RectangularMatrix)
{
    int rows = 2, cols = 3;
    cuDoubleComplex input[] = {
        make_cuDoubleComplex(1.0, 0.1), make_cuDoubleComplex(2.0, 0.2), make_cuDoubleComplex(3.0, 0.3),
        make_cuDoubleComplex(4.0, 0.4), make_cuDoubleComplex(5.0, 0.5), make_cuDoubleComplex(6.0, 0.6)};
    cuDoubleComplex expected[] = {
        make_cuDoubleComplex(1.0, 0.1), make_cuDoubleComplex(4.0, 0.4),
        make_cuDoubleComplex(2.0, 0.2), make_cuDoubleComplex(5.0, 0.5),
        make_cuDoubleComplex(3.0, 0.3), make_cuDoubleComplex(6.0, 0.6)};
    cuDoubleComplex output[6];

    Transpose(input, output, rows, cols);

    ASSERT_TRUE(compareCuDoubleComplexArrays(output, expected, rows * cols));
}

// Test case for an empty matrix
TEST(TransposeCuDoubleComplexTest, EmptyMatrix)
{
    int rows = 0, cols = 0;
    cuDoubleComplex *input = nullptr;
    cuDoubleComplex *output = nullptr;

    Transpose(input, output, rows, cols);

    ASSERT_TRUE(output == nullptr);
}

// Test case for a square matrix
TEST(TransposeCuDoubleComplexInPlaceTest, SquareMatrix)
{
    int rows = 3, cols = 3;
    cuDoubleComplex matrix[] = {
        make_cuDoubleComplex(1.0, 0.1), make_cuDoubleComplex(2.0, 0.2), make_cuDoubleComplex(3.0, 0.3),
        make_cuDoubleComplex(4.0, 0.4), make_cuDoubleComplex(5.0, 0.5), make_cuDoubleComplex(6.0, 0.6),
        make_cuDoubleComplex(7.0, 0.7), make_cuDoubleComplex(8.0, 0.8), make_cuDoubleComplex(9.0, 0.9)};
    cuDoubleComplex expected[] = {
        make_cuDoubleComplex(1.0, 0.1), make_cuDoubleComplex(4.0, 0.4), make_cuDoubleComplex(7.0, 0.7),
        make_cuDoubleComplex(2.0, 0.2), make_cuDoubleComplex(5.0, 0.5), make_cuDoubleComplex(8.0, 0.8),
        make_cuDoubleComplex(3.0, 0.3), make_cuDoubleComplex(6.0, 0.6), make_cuDoubleComplex(9.0, 0.9)};

    Transpose(matrix, rows, cols);

    ASSERT_TRUE(compareCuDoubleComplexArrays(matrix, expected, rows * cols));
}

// Test case for a non-square matrix (should throw an exception)
TEST(TransposeCuDoubleComplexInPlaceTest, NonSquareMatrix)
{
    int rows = 2, cols = 3;
    cuDoubleComplex matrix[] = {
        make_cuDoubleComplex(1.0, 0.1), make_cuDoubleComplex(2.0, 0.2), make_cuDoubleComplex(3.0, 0.3),
        make_cuDoubleComplex(4.0, 0.4), make_cuDoubleComplex(5.0, 0.5), make_cuDoubleComplex(6.0, 0.6)};

    EXPECT_THROW(Transpose(matrix, rows, cols), std::invalid_argument);
}

// Test case for a 1x1 matrix
TEST(TransposeCuDoubleComplexInPlaceTest, SingleElementMatrix)
{
    int rows = 1, cols = 1;
    cuDoubleComplex matrix[] = {make_cuDoubleComplex(1.0, 0.1)};
    cuDoubleComplex expected[] = {make_cuDoubleComplex(1.0, 0.1)};

    Transpose(matrix, rows, cols);

    ASSERT_TRUE(compareCuDoubleComplexArrays(matrix, expected, rows * cols));
}