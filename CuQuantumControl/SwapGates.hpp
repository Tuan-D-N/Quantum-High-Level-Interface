#pragma once
#include "Precision.hpp"

template <precision SelectPrecision = precision::bit_64>
int swap(custatevecHandle_t &handle,
         const int nIndexBits,
         const int2 bitSwaps[],
         const int nBitSwaps,
         PRECISION_TYPE_COMPLEX(SelectPrecision) *d_sv);