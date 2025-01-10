#include <vector>
#include <cuComplex.h>
#include "Transpose.hpp"
#include <stdexcept>

// Function to transpose a 2D matrix
std::vector<std::vector<int>> Transpose(const std::vector<std::vector<int>> &matrix)
{
    if (matrix.empty())
        return {};

    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<int>> result(cols, std::vector<int>(rows));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}
void Transpose(const cuDoubleComplex *input, cuDoubleComplex *output, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

void Transpose(cuDoubleComplex *matrix, int rows, int cols)
{
    if (rows != cols)
    {
        throw std::invalid_argument("In-place transpose requires a square matrix.");
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = i + 1; j < cols; ++j)
        {
            // Swap elements at (i, j) and (j, i)
            cuDoubleComplex temp = matrix[i * cols + j];
            matrix[i * cols + j] = matrix[j * cols + i];
            matrix[j * cols + i] = temp;
        }
    }
}