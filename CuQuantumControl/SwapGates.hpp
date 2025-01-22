#pragma once
#include <span>
#include <cuComplex.h>
#include <custatevec.h>
#include "Precision.hpp"
#include "../CudaControl/Helper.hpp"

/// @brief applySwap the qubits data
/// @tparam SelectPrecision Precision
/// @param handle the handle obj
/// @param nIndexBits number of qubtis
/// @param bitSwaps array of array to applySwap bits, Swap qubit 1 <-> 2 and 3 <-> 4: {{1,2},{3,4}}. Cannot overlap in qubits
/// @param nBitSwaps number of swaps that there are
/// @param d_sv input statevector
/// @return
template <precision SelectPrecision = precision::bit_64>
int applySwap(custatevecHandle_t &handle,
         const int nIndexBits,
         const int2 bitSwaps[],
         const int nBitSwaps,
         PRECISION_TYPE_COMPLEX(SelectPrecision) * d_sv);

/// @brief applySwap the qubits data
/// @tparam SelectPrecision Precision
/// @param handle the handle obj
/// @param nIndexBits number of qubtis
/// @param bitSwap span object of qubits to applySwap, Swap qubit 1 <-> 2 and 3 <-> 4: {{1,2},{3,4}}. Cannot overlap in qubits
/// @param d_sv input statevector
/// @return 
template <precision SelectPrecision = precision::bit_64>
int applySwap(custatevecHandle_t &handle,
         const int nIndexBits,
         const std::span<const int2> bitSwap,
         PRECISION_TYPE_COMPLEX(SelectPrecision) * d_sv)
{
    CHECK_BROAD_ERROR(applySwap<SelectPrecision>(
        handle,
        nIndexBits,
        bitSwap.data(),
        bitSwap.size(),
        d_sv));
    return cudaSuccess;
}