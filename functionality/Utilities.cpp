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

template <Streamable T>
void print2DVector(const std::vector<std::vector<T>> &vec)
{
    for (const auto &row : vec)
    { // Iterate over each row
        for (T val : row)
        {                            // Iterate over each column in the row
            std::cout << val << " "; // Print the element
        }
        std::cout << "\n"; // Move to the next line after printing a row
    }
}

template void print2DVector<double>(const std::vector<std::vector<double>> &vec);
template void print2DVector<int>(const std::vector<std::vector<int>> &vec);
template void print2DVector<float>(const std::vector<std::vector<float>> &vec);
template void print2DVector<std::complex<double>>(const std::vector<std::vector<std::complex<double>>> &vec);
template void print2DVector<std::complex<int>>(const std::vector<std::vector<std::complex<int>>> &vec);
template void print2DVector<std::complex<float>>(const std::vector<std::vector<std::complex<float>>> &vec);

template <Streamable T>
void printVector(const std::vector<T> &vec)
{
    for (T val : vec)
    { // Iterate over the vector and print each element
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

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

bool almost_equal(cuDoubleComplex x, cuDoubleComplex y) {
    const double eps = 1.0e-5;
    const cuDoubleComplex diff = cuCsub(x, y);
    return (cuCabs(diff) < eps);
}

bool almost_equal(double x, double y) {
    const double eps = 1.0e-5;
    const double diff = x - y;
    return (abs(diff) < eps);
}