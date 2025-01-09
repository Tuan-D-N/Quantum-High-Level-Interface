#include "fftShift.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <cuComplex.h>

void fftshift1D(cuDoubleComplex* data, int n) {
    int half = n / 2;
    if (n % 2 == 0) {
        std::rotate(data, data + half, data + n);
    } else {
        std::rotate(data, data + half + 1, data + n);
    }
}

void fftshift1D(std::vector<double>& data) {
    int n = data.size();
    int half = n / 2;
    if (n % 2 == 0) {
        std::rotate(data.begin(), data.begin() + half, data.end());
    } else {
        std::rotate(data.begin(), data.begin() + half + 1, data.end());
    }
}


void fftshift1D(double* data, int length) {
    int half = length / 2;
    if (length % 2 == 0) {
        std::rotate(data, data + half, data + length);
    } else {
        std::rotate(data, data + half + 1, data + length);
    }
}

void fftshift2D(cuDoubleComplex* data, int rows, int cols) {
    // Shift each row
    for (int i = 0; i < rows; ++i) {
        fftshift1D(data + i * cols, cols);
    }

    // Shift each column
    for (int j = 0; j < cols; ++j) {
        // Extract column into temporary storage
        std::vector<cuDoubleComplex> column(rows);
        for (int i = 0; i < rows; ++i) {
            column[i] = data[i * cols + j];
        }

        // Apply 1D fftshift to the column
        fftshift1D(column.data(), rows);

        // Write back the shifted column
        for (int i = 0; i < rows; ++i) {
            data[i * cols + j] = column[i];
        }
    }
}