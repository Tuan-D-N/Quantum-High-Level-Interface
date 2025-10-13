#pragma once
#include <vector>
#include <span>
#include <algorithm>
#include <cuComplex.h>

void fftshift1D(std::vector<double> &data);

void fftshift1D(double *data, int length);

void fftshift1D(cuDoubleComplex *data, int length);

void fftshift2D(cuDoubleComplex *data, int rows, int cols);

template <typename T>
void fftshift1D(T *data, int length)
{
    int half = length / 2;
    if (length % 2 == 0)
    {
        std::rotate(data, data + half, data + length);
    }
    else
    {
        std::rotate(data, data + half + 1, data + length);
    }
}

template <typename T>
void fftshift1D(T *data, int size, int stride)
{
    throw std::runtime_error("ERROR: not implemented.");
    // int half = size / 2;
    // for (int i = 0; i < half; ++i)
    // {
    //     std::swap(data[i * stride], data[(i + half) * stride]);
    // }
}

template <typename T>
void fftshift1D(std::span<T> data)
{
    fftshift1D<T>(data.data(), data.size());
}

template <typename T>
void fftshift1D(std::span<T> data, int stride)
{
    fftshift1D<T>(data.data(), data.size(), stride);
}

template <typename T>
void fftshift2D(T *data, int rows, int cols)
{
// Shift each row
#pragma omp parallel for
    for (int i = 0; i < rows; ++i)
    {
        fftshift1D(data + i * cols, cols);
    }

// Shift each column
#pragma omp parallel for
    for (int j = 0; j < cols; ++j)
    {
        // Extract column into temporary storage
        std::vector<T> column(rows);
        for (int i = 0; i < rows; ++i)
        {
            column[i] = data[i * cols + j];
        }

        // Apply 1D fftshift to the column
        fftshift1D(column.data(), rows);

        // Write back the shifted column
        for (int i = 0; i < rows; ++i)
        {
            data[i * cols + j] = column[i];
        }
    }
}