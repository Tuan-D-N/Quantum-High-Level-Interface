#include "Utilities.hpp"
#include <vector>
#include <iostream>
#include <complex>
#include <cuComplex.h>

bool isEven(int num)
{
    return num % 2 == 0;
}

bool isOdd(int num)
{
    return num % 2 != 0;
}


template void print2DVector<double>(const std::vector<std::vector<double>> &vec);
template void print2DVector<int>(const std::vector<std::vector<int>> &vec);
template void print2DVector<float>(const std::vector<std::vector<float>> &vec);
template void print2DVector<std::complex<double>>(const std::vector<std::vector<std::complex<double>>> &vec);
template void print2DVector<std::complex<int>>(const std::vector<std::vector<std::complex<int>>> &vec);
template void print2DVector<std::complex<float>>(const std::vector<std::vector<std::complex<float>>> &vec);



template void printVector<double>(const std::vector<double> &vec);
template void printVector<int>(const std::vector<int> &vec);
template void printVector<float>(const std::vector<float> &vec);
template void printVector<std::complex<double>>(const std::vector<std::complex<double>> &vec);
template void printVector<std::complex<int>>(const std::vector<std::complex<int>> &vec);
template void printVector<std::complex<float>>(const std::vector<std::complex<float>> &vec);

std::vector<std::vector<double>> csrToDense(
    const cuDoubleComplex *values,  // Non-zero values
    const std::vector<int> &rowPtr, // Row pointers
    const std::vector<int> &cols,   // Column indices
    int rows,                       // Number of rows
    int colsCount                   // Number of columns
)
{
    // Initialize a dense matrix with zeros
    std::vector<std::vector<double>> dense(rows, std::vector<double>(colsCount, 0));

    // Iterate through each row
    for (int i = 0; i < rows; ++i)
    {
        // Non-zero elements for the row are in the range [rowPtr[i], rowPtr[i + 1])
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
        {
            dense[i][cols[j]] = cuCreal(values[j]);
        }
    }
    return dense;
}

std::vector<std::vector<double>> cscToDense(
    const cuDoubleComplex *values,  // Non-zero values
    const std::vector<int> &colPtr, // Column pointers
    const std::vector<int> &rows,   // Row indices
    int rowsCount,                  // Number of rows
    int colsCount                   // Number of columns
)
{
    // Initialize a dense matrix with zeros
    std::vector<std::vector<double>> dense(rowsCount, std::vector<double>(colsCount, 0));

    // Iterate through each column
    for (int j = 0; j < colsCount; ++j)
    {
        // Non-zero elements for the column are in the range [colPtr[j], colPtr[j + 1])
        for (int i = colPtr[j]; i < colPtr[j + 1]; ++i)
        {
            dense[rows[i]][j] = cuCreal(values[i]);
        }
    }

    return dense;
}

void printDeviceArray(cuDoubleComplex *d_array, int size)
{
    cuDoubleComplex *h_array = new cuDoubleComplex[size];
    cudaMemcpy(h_array, d_array, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i)
    {
        std::cout << "(" << h_array[i].x << ", " << h_array[i].y << ") ";
        if (i != size - 1)
        {
            std::cout << ",";
        }
    }
    std::cout << std::endl;

    delete[] h_array;
}

void printDeviceArray(cuFloatComplex *d_array, int size)
{
    cuFloatComplex *h_array = new cuFloatComplex[size];
    cudaMemcpy(h_array, d_array, size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i)
    {
        std::cout << "(" << h_array[i].x << ", " << h_array[i].y << ") ";
        if (i != size - 1)
        {
            std::cout << ",";
        }
    }
    std::cout << std::endl;

    delete[] h_array;
}

bool almost_equal(cuDoubleComplex x, cuDoubleComplex y) {
    const double eps = 1.0e-5;
    const cuDoubleComplex diff = cuCsub(x, y);
    return (cuCabs(diff) < eps);
}

bool almost_equal(cuFloatComplex x, cuFloatComplex y) {
    const double eps = 1.0e-5;
    const cuFloatComplex diff = cuCsubf(x, y);
    return (cuCabsf(diff) < eps);
}

bool almost_equal(double x, double y) {
    const double eps = 1.0e-5;
    const double diff = x - y;
    return (abs(diff) < eps);
}

std::vector<int> rangeVec(int start, int end)
{
    std::vector<int> vec(end - start);
    std::iota(vec.begin(), vec.end(), start);
    return vec;
}

bool isPowerOf2(int num) {
    return num > 0 && (num & (num - 1)) == 0;
}

bool are_disjoint(std::span<const int> listA, std::span<const int> listB) {
    for (const auto& elementA : listA) {
        if (std::find(listB.begin(), listB.end(), elementA) != listB.end()) {
            return false; // Overlap found
        }
    }
    return true; // No overlap
}
