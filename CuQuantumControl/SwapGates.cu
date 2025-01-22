#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <bitset>
#include "../CudaControl/Helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR
#include "SwapGates.hpp"
#include "Precision.hpp"

template <precision SelectPrecision>
int applySwap(custatevecHandle_t &handle,
         const int nIndexBits,
         const int2 bitSwaps[],
         const int nBitSwaps,
         PRECISION_TYPE_COMPLEX(SelectPrecision) *d_sv)

{
    cudaDataType_t cudaType;
    custatevecComputeType_t custatevecType;
    if constexpr (SelectPrecision == precision::bit_32)
    {
        cudaType = CUDA_C_32F;
        custatevecType = CUSTATEVEC_COMPUTE_32F;
    }
    else if constexpr (SelectPrecision == precision::bit_64)
    {
        cudaType = CUDA_C_64F;
        custatevecType = CUSTATEVEC_COMPUTE_64F;
    }

    // applySwap the state vector elements only if 1st qubit is 1
    const int maskLen = 0;
    int maskBitString[] = {};
    int maskOrdering[] = {};
    // bit applySwap
    CHECK_CUSTATEVECTOR(custatevecSwapIndexBits(
        handle, d_sv, cudaType, nIndexBits, bitSwaps, nBitSwaps,
        maskBitString, maskOrdering, maskLen));

    return cudaSuccess;
}

template int applySwap<precision::bit_32>(custatevecHandle_t &handle,
         const int nIndexBits,
         const int2 bitSwaps[],
         const int nBitSwaps,
         PRECISION_TYPE_COMPLEX(precision::bit_32) *d_sv);

template int applySwap<precision::bit_64>(custatevecHandle_t &handle,
         const int nIndexBits,
         const int2 bitSwaps[],
         const int nBitSwaps,
         PRECISION_TYPE_COMPLEX(precision::bit_64) *d_sv);