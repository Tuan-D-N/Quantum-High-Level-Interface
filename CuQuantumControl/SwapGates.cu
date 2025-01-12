#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <bitset>
#include "../CudaControl/Helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR
#include "SwapGates.hpp"

int swap(custatevecHandle_t &handle,
         const int nIndexBits,
         const int2 bitSwaps[],
         const int nBitSwaps,
         cuDoubleComplex *d_sv)

{

    // swap the state vector elements only if 1st qubit is 1
    const int maskLen = 0;
    int maskBitString[] = {};
    int maskOrdering[] = {};
    // bit swap
    CHECK_CUSTATEVECTOR(custatevecSwapIndexBits(
        handle, d_sv, CUDA_C_64F, nIndexBits, bitSwaps, nBitSwaps,
        maskBitString, maskOrdering, maskLen));

    return cudaSuccess;
}