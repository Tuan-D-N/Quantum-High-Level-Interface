#pragma once
#include <custatevec.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <iostream>
#include <span>
#include <vector>
#include <cassert>

#include "../CudaControl/Helper.hpp"
#include "Precision.hpp"

/// @brief This function generates a substatevector of a statevector using masking of certain qubits
/// @tparam SelectPrecision
/// @param handle handle obj
/// @param nIndexBits number of qubits
/// @param bitOrderingLen Length of bit ordering array
/// @param bitOrdering The order of the bits to output in an cArray. 1st qubit will be the fastest changing qubit in the new state vector
/// @param maskLen length of the maskBitString and maskOrdering.
/// @param maskBitString A cArray of {0,1} values determining which value will be determined to mask.
/// @param maskOrdering A cArray of [0,nIndexBits) values determining the position of the bit to mask.
/// @param d_sv device array.
/// @param buffer_access_begin Inclusive value. Determines the starting index of the buffer. Can be used to exclude the first few values if we know that they are small and not useful.
/// @param buffer_access_end Exclusive value. Determines the ending index of the buffer.
/// @param out_buffer A cArray of the resulting sub-statevector.
/// @param extraWorkspace workspace pointer
/// @param extraWorkspaceSizeInBytes size of workspace pointer. Note workspace may get larger.
/// @return (int) error value
template <precision SelectPrecision = precision::bit_64>
int applyAccessorGet(custatevecHandle_t &handle,
                const int nIndexBits,
                const int bitOrderingLen,
                const int bitOrdering[],
                const int maskLen,
                const int maskBitString[],
                const int maskOrdering[],
                PRECISION_TYPE_COMPLEX(SelectPrecision) * d_sv,
                const int buffer_access_begin,
                const int buffer_access_end,
                PRECISION_TYPE_COMPLEX(SelectPrecision) * out_buffer,
                void *extraWorkspace,
                size_t &extraWorkspaceSizeInBytes);

template <precision SelectPrecision = precision::bit_64>
int applyAccessorGet(custatevecHandle_t &handle,
                const int nIndexBits,
                std::span<const int> bitOrdering,
                std::span<const int> maskBitString,
                std::span<const int> maskOrdering,
                PRECISION_TYPE_COMPLEX(SelectPrecision) * d_sv,
                std::vector<PRECISION_TYPE_COMPLEX(SelectPrecision)> &out_buffer,
                const int buffer_start,
                const int buffer_end,
                void *extraWorkspace,
                size_t &extraWorkspaceSizeInBytes)
{
    assert(maskBitString.size() == maskOrdering.size());
    const int bufferSize = buffer_end - buffer_start;
    out_buffer.reserve(bufferSize);
    CHECK_BROAD_ERROR(applyAccessorGet(handle,
                                  nIndexBits,
                                  bitOrdering.size(),
                                  bitOrdering.data(),
                                  maskBitString.size(),
                                  maskBitString.data(),
                                  maskOrdering.data(),
                                  d_sv,
                                  buffer_start,
                                  buffer_end,
                                  out_buffer.data(),
                                  extraWorkspace,
                                  extraWorkspaceSizeInBytes));
    return cudaSuccess;
}

template <precision SelectPrecision = precision::bit_64>
int applyAccessorGet(custatevecHandle_t &handle,
                const int nIndexBits,
                std::span<const int> bitOrdering,
                std::span<const int> maskBitString,
                std::span<const int> maskOrdering,
                PRECISION_TYPE_COMPLEX(SelectPrecision) * d_sv,
                std::vector<PRECISION_TYPE_COMPLEX(SelectPrecision)> &out_buffer,
                void *extraWorkspace,
                size_t &extraWorkspaceSizeInBytes)
{
    assert(maskBitString.size() == maskOrdering.size());
    const int bufferSize = 2 << (nIndexBits - maskOrdering.size());
    const int buffer_start = 0;
    const int buffer_end = bufferSize;
    CHECK_BROAD_ERROR(applyAccessorGet(handle,
                                  nIndexBits,
                                  bitOrdering,
                                  maskBitString,
                                  maskOrdering,
                                  d_sv,
                                  out_buffer,
                                  buffer_start,
                                  buffer_end,
                                  extraWorkspace,
                                  extraWorkspaceSizeInBytes));
    return cudaSuccess;
}