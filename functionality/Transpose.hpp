#pragma once
#include <vector>
#include <cuComplex.h>

std::vector<std::vector<int>> Transpose(const std::vector<std::vector<int>> &matrix);

void Transpose(const cuDoubleComplex *input, cuDoubleComplex *output, int rows, int cols);

void Transpose(cuDoubleComplex *matrix, int rows, int cols);