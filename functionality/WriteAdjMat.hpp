#pragma once
#include <array>
#include <vector>
#include <cuComplex.h>




int getRowOffsetSizeMini(int evenQubits);
int getColumnIndexSizeMini(int evenQubits);
int getValuesSizeMini(int evenQubits);


void writeMatAMiniCSR(int* rowOffset, int* columnIndex, cuDoubleComplex* values, int evenQubits, int &rowOffsetSize, int &columnIndexSize, int &valuesSize);

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
