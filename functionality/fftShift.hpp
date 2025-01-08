#pragma once
#include <vector>
#include <algorithm>
#include <cuComplex.h>

void fftshift1D(std::vector<double>& data);

void fftshift1D(double* data, int length);

void fftshift1D(cuDoubleComplex* data, int length);