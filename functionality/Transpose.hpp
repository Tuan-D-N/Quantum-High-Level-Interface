#pragma once
#include <vector>
#include <cuComplex.h>
#include <stdexcept>


std::vector<std::vector<int>> Transpose(const std::vector<std::vector<int>> &matrix);

void Transpose(const cuDoubleComplex *input, cuDoubleComplex *output, int rows, int cols);

void Transpose(cuDoubleComplex *matrix, int rows, int cols);


// Function to transpose a 2D matrix
template<typename T>
std::vector<std::vector<T>> Transpose(const std::vector<std::vector<T>> &matrix)
{
    if (matrix.empty())
        return {};

    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<T>> result(cols, std::vector<T>(rows));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

template <typename T>
void Transpose(const T *input, T *output, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

template <typename T>
void Transpose(T *matrix, int rows, int cols)
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
            T temp = matrix[i * cols + j];
            matrix[i * cols + j] = matrix[j * cols + i];
            matrix[j * cols + i] = temp;
        }
    }
}