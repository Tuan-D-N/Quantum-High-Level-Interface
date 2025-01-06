#pragma once
#include <array>
#include <cuComplex.h>




int getRowOffsetSizeMini(int evenQubits);
int getColumnIndexSizeMini(int evenQubits);
int getValuesSizeMini(int evenQubits);


void writeMatAMiniCSR(int* rowOffset, int* columnIndex, cuDoubleComplex* values, int evenQubits, int &rowOffsetSize, int &columnIndexSize, int &valuesSize);