#include <custatevec.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <iostream>

#include "Accessor.hpp"
#include "../CudaControl/Helper.hpp"
#include "Precision.hpp"

template <precision SelectPrecision>
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
                     void *&extraWorkspace,
                     size_t &extraWorkspaceSizeInBytes)
{
    // Precision generate -----------------------------------------------------
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
    // Precision generate -----------------------------------------------------

    size_t extraWorkspaceSizeInBytes_CHECK = extraWorkspaceSizeInBytes;
    custatevecAccessorDescriptor_t accessor;
    CHECK_CUSTATEVECTOR(custatevecAccessorCreateView(
        handle, d_sv, cudaType, nIndexBits, &accessor, bitOrdering, bitOrderingLen,
        maskBitString, maskOrdering, maskLen, &extraWorkspaceSizeInBytes_CHECK));

    if (extraWorkspaceSizeInBytes_CHECK > extraWorkspaceSizeInBytes)
    {
        std::cout << "Extra space needed: " << extraWorkspaceSizeInBytes_CHECK - extraWorkspaceSizeInBytes << " Bytes\n";
        if (extraWorkspace != nullptr)
        {
            CHECK_CUDA(cudaFree(extraWorkspace));
        }
        CHECK_CUDA(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes_CHECK));
    }

    // set external workspace
    CHECK_CUSTATEVECTOR(custatevecAccessorSetExtraWorkspace(
        handle, accessor, extraWorkspace, extraWorkspaceSizeInBytes_CHECK));

    // get state vector components
    CHECK_CUSTATEVECTOR(custatevecAccessorGet(
        handle, accessor, out_buffer, buffer_access_begin, buffer_access_end));

    CHECK_CUSTATEVECTOR(custatevecAccessorDestroy(accessor));
    THROW_CUDA(cudaDeviceSynchronize());
    return cudaSuccess;
}

template int applyAccessorGet<precision::bit_32>(custatevecHandle_t &handle,
                                                 const int nIndexBits,
                                                 const int bitOrderingLen,
                                                 const int bitOrdering[],
                                                 const int maskLen,
                                                 const int maskBitString[],
                                                 const int maskOrdering[],
                                                 PRECISION_TYPE_COMPLEX(precision::bit_32) * d_sv,
                                                 const int buffer_access_begin,
                                                 const int buffer_access_end,
                                                 PRECISION_TYPE_COMPLEX(precision::bit_32) * out_buffer,
                                                 void *&extraWorkspace,
                                                 size_t &extraWorkspaceSizeInBytes);

template int applyAccessorGet<precision::bit_64>(custatevecHandle_t &handle,
                                                 const int nIndexBits,
                                                 const int bitOrderingLen,
                                                 const int bitOrdering[],
                                                 const int maskLen,
                                                 const int maskBitString[],
                                                 const int maskOrdering[],
                                                 PRECISION_TYPE_COMPLEX(precision::bit_64) * d_sv,
                                                 const int buffer_access_begin,
                                                 const int buffer_access_end,
                                                 PRECISION_TYPE_COMPLEX(precision::bit_64) * out_buffer,
                                                 void *&extraWorkspace,
                                                 size_t &extraWorkspaceSizeInBytes);