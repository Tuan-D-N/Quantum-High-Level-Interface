#include <array>
#include <cuComplex.h>

using complex = cuDoubleComplex;


int getRowOffsetSizeMini(int evenQubits);
int getColumnIndexSizeMini(int evenQubits);
int getValuesSizeMini(int evenQubits);


void writeMatAMiniCSR(int* rowOffset, int* columnIndex, complex* values, int evenQubits, int &rowOffsetSize, int &columnIndexSize, int &valuesSize);