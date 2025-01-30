#pragma once
#include <vector>
#include <span>
#include <complex>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>

/// @brief The function speaks for itself
/// @param num integer only
/// @return true/false
bool isOdd(int num);

/// @brief Are you literally dumb to read this
/// @param num integer only
/// @return true/false
bool isEven(int num);

// Define a concept for types that can be used with std::ostream
template <typename T>
concept Streamable = requires(std::ostream &os, T value) {
    { os << value } -> std::same_as<std::ostream &>; // Ensures os << value is valid and returns std::ostream&
};


/// @brief Print a 2D vector out
/// @tparam T a streamable type
/// @param vec input vector
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

/// @brief print a 1D vector out
/// @tparam T a streamable type
/// @param vec input vector
template <Streamable T>
void printVector(const std::vector<T> &vec)
{
    for (T val : vec)
    { // Iterate over the vector and print each element
        std::cout << val << " ";
    }
    std::cout << std::endl;
}



template <typename T>
int sign(T value)
{
    if (value == 0)
        return 0;
    return std::signbit(value) ? -1 : 1;
}

std::vector<std::vector<double>> csrToDense(
    const cuDoubleComplex *values,  // Non-zero values
    const std::vector<int> &rowPtr, // Row pointers
    const std::vector<int> &cols,   // Column indices
    int rows,                       // Number of rows
    int colsCount                   // Number of columns
);

std::vector<std::vector<double>> cscToDense(
    const cuDoubleComplex *values,  // Non-zero values
    const std::vector<int> &colPtr, // Column pointers
    const std::vector<int> &rows,   // Row indices
    int rowsCount,                  // Number of rows
    int colsCount                   // Number of columns
);

template <typename T>
std::vector<std::vector<T>> csrToDense(
    const T *values,                // Non-zero values
    const std::vector<int> &rowPtr, // Row pointers
    const std::vector<int> &cols,   // Column indices
    int rows,                       // Number of rows
    int colsCount                   // Number of columns
)
{
    // Initialize a dense matrix with zeros
    std::vector<std::vector<T>> dense(rows, std::vector<T>(colsCount, 0));

    // Iterate through each row
    for (int i = 0; i < rows; ++i)
    {
        // Non-zero elements for the row are in the range [rowPtr[i], rowPtr[i + 1])
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
        {
            dense[i][cols[j]] = (values[j]);
        }
    }
    return dense;
}

template <typename T>
std::vector<std::vector<T>> cscToDense(
    const T *values,                // Non-zero values
    const std::vector<int> &colPtr, // Column pointers
    const std::vector<int> &rows,   // Row indices
    int rowsCount,                  // Number of rows
    int colsCount                   // Number of columns
)
{
    // Initialize a dense matrix with zeros
    std::vector<std::vector<T>> dense(rowsCount, std::vector<T>(colsCount, 0));

    // Iterate through each column
    for (int j = 0; j < colsCount; ++j)
    {
        // Non-zero elements for the column are in the range [colPtr[j], colPtr[j + 1])
        for (int i = colPtr[j]; i < colPtr[j + 1]; ++i)
        {
            dense[rows[i]][j] = (values[i]);
        }
    }

    return dense;
}

// Helper to get default tolerance based on type
template <typename T>
constexpr T getDefaultToleranceForMatricesEqual() {
    if constexpr (std::is_floating_point_v<T>) {
        return static_cast<T>(1e-5); // Default for floating-point types
    } else {
        return static_cast<T>(0);    // Default for other types
    }
}

template <typename T>
bool matricesEqual(const std::vector<std::vector<T>> &matrix1, const std::vector<std::vector<T>> &matrix2, T tolerance = getDefaultToleranceForMatricesEqual<T>())
{
    if (matrix1.size() != matrix2.size())
    {
        return false;
    }
    else if (matrix1.size() == 0)
    {
        return true;
    }
    else if (matrix1[0].size() != matrix2[0].size())
    {
        return false;
    }

    for (size_t i = 0; i < matrix1.size(); ++i)
    {
        for (size_t j = 0; j < matrix1[0].size(); ++j)
        {
            if (abs(matrix1[i][j] - matrix2[i][j]) > tolerance)
            {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
void printDeviceArray(T *d_array, int size)
{
    T *h_array = new T[size];
    cudaMemcpy(h_array, d_array, size * sizeof(T), cudaMemcpyDeviceToHost);
    for (T i = 0; i < size; ++i)
        std::cout << h_array[i] << " ";
    std::cout << std::endl;
    delete[] h_array;
}

void printDeviceArray(cuDoubleComplex *d_array, int size);
void printDeviceArray(cuFloatComplex *d_array, int size);

bool almost_equal(cuDoubleComplex x, cuDoubleComplex y);
bool almost_equal(cuFloatComplex x, cuFloatComplex y);

bool almost_equal(double x, double y);

std::vector<int> rangeVec(int start, int end);

bool isPowerOf2(int num);

bool are_disjoint(std::span<const int> listA, std::span<const int> listB);