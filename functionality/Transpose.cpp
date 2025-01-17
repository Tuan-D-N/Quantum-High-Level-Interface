#include <vector>
#include <cuComplex.h>
#include "Transpose.hpp"
#include <stdexcept>

// Function to transpose a 2D matrix
std::vector<std::vector<int>> Transpose(const std::vector<std::vector<int>> &matrix)
{
        return Transpose<int>(matrix);
}
void Transpose(const cuDoubleComplex *input, cuDoubleComplex *output, int rows, int cols)
{
    Transpose<cuDoubleComplex>(input, output, rows, cols);
}

void Transpose(cuDoubleComplex *matrix, int rows, int cols)
{
    Transpose<cuDoubleComplex>(matrix, rows, cols);
}