#pragma once
#include <array>
#include <vector>
#include <cuComplex.h>




int getRowOffsetSizeMini(int evenQubits);
int getColumnIndexSizeMini(int evenQubits);
int getValuesSizeMini(int evenQubits);


void writeMatAMiniCSC(int* ColumnOffset, int* rowIndex, cuDoubleComplex* values, int evenQubits, int &ColumnOffsetSize, int &rowIndexSize, int &valuesSize);

std::vector<std::vector<double>> csrToDense(
    const cuDoubleComplex *values,  // Non-zero values
    const std::vector<int> &rowPtr, // Row pointers
    const std::vector<int> &cols,   // Column indices
    int rows,                       // Number of rows
    int colsCount                   // Number of columns
);
