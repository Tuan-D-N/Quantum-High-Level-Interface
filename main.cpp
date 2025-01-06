#include <iostream>
#include <cuComplex.h>
#include "functionality/WriteAdjMat.hpp"
#include "functionality/Utilities.hpp"

int main()
{
    int evenqubits = 6;
    int *rowOffset = new int[getRowOffsetSizeMini(evenqubits)];
    int *columnIndices = new int[getColumnIndexSizeMini(evenqubits)];
    complex *values = new complex[getValuesSizeMini(evenqubits)];

    int rowOffsetSize;
    int columnIndicesSize;
    int valuesSize;

    writeMatACSR(rowOffset,
                 columnIndices,
                 values,
                 evenqubits,
                 rowOffsetSize,
                 columnIndicesSize,
                 valuesSize);

    for (int i = 0; i < rowOffsetSize; ++i)
    {
        std::cout << rowOffset[i] << " , ";
    }
    std::cout << std::endl;

    for (int i = 0; i < columnIndicesSize; ++i)
    {
        std::cout << columnIndices[i] << " , ";
    }
    std::cout << std::endl;

    for (int i = 0; i < valuesSize; ++i)
    {
        std::cout << cuCreal(values[i]) << " , ";
    }
    std::cout << std::endl;

    delete[] rowOffset;
    delete[] columnIndices;
    delete[] values;

    return 0;
}
