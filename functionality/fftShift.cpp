#include "fftShift.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <cuComplex.h>

void fftshift1D(cuDoubleComplex *data, int n)
{
    fftshift1D<cuDoubleComplex>(data, n);
}

void fftshift1D(std::vector<double> &data)
{
    fftshift1D<double>(data);
}

void fftshift1D(double *data, int length)
{
    fftshift1D<double>(data, length);
}

void fftshift2D(cuDoubleComplex *data, int rows, int cols)
{
    fftshift2D<cuDoubleComplex>(data, rows, cols);
}