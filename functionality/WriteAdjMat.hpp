#pragma once
#include <array>
#include <vector>
#include <cuComplex.h>




int getRowOffsetSizeMini(int evenQubits);
int getColumnIndexSizeMini(int evenQubits);
int getValuesSizeMini(int evenQubits);


void writeMatAMiniCSC(int* ColumnOffset, int* rowIndex, cuDoubleComplex* values, int evenQubits, int &ColumnOffsetSize, int &rowIndexSize, int &valuesSize, bool mask = false);

