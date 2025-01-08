#pragma once

int swap(custatevecHandle_t &handle,
         const int nIndexBits,
         const int2 bitSwaps[],
         const int nBitSwaps,
         cuDoubleComplex *d_sv);