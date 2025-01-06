#include <array>
#include <cuComplex.h>

using complex = cuDoubleComplex;


int getRowOffsetSize(int evenQubits);
int getColumnIndexSize(int evenQubits);
int getValuesSize(int evenQubits);


void writeMatACSR(int* rowOffset, int* columnIndex, complex* values, int evenQubits, int &rowOffsetSize, int &columnIndexSize, int &valuesSize);