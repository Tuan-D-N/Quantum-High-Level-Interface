#pragma once
#include <vector>
#include <algorithm>

void fftshift1D(std::vector<double>& data);

void fftshift1D(double* data, int length);