#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <custatevec.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "../CudaControl/Helper.hpp"

#include "../functionality/randomArray.hpp"
#include "ApplySampler.hpp"
#include "Precision.hpp"

template <precision SelectPrecision>
int sampleSV(custatevecHandle_t &handle,
             const int nIndexBits,
             const int bitOrdering[], // Qubits to measure
             const int bitStringLen,  // length of bitOrdering
             PRECISION_TYPE_COMPLEX(SelectPrecision) d_sv[],
             custatevecIndex_t bitStrings_out[],
             const int nShots,
             void *extraWorkspace,
             size_t &extraWorkspaceSizeInBytes,
             double randnums[])
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

    size_t extraWorkspaceSizeInBytes_CHECK = extraWorkspaceSizeInBytes;
    int nMaxShots = nShots;

    custatevecSamplerDescriptor_t sampler;
    // create sampler and check the size of external workspace
    CHECK_CUSTATEVECTOR(custatevecSamplerCreate(
        handle, d_sv, cudaType, nIndexBits, &sampler, nMaxShots,
        &extraWorkspaceSizeInBytes_CHECK));

    if (extraWorkspaceSizeInBytes_CHECK > extraWorkspaceSizeInBytes)
    {
        std::cout << "Extra space needed: " << extraWorkspaceSizeInBytes_CHECK - extraWorkspaceSizeInBytes << " Bytes\n";
        if (extraWorkspace != nullptr)
        {
            CHECK_CUDA(cudaFree(extraWorkspace));
        }
        CHECK_CUDA(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes_CHECK));
    }

    // sample preprocess
    CHECK_CUSTATEVECTOR(custatevecSamplerPreprocess(
        handle, sampler, extraWorkspace, extraWorkspaceSizeInBytes_CHECK));

    // In real appliction, random numbers in range [0, 1) will be used.
    bool freeRandnums = false;
    if (randnums == nullptr)
    {
        freeRandnums = true;
        randnums = new double[nShots];
        generateRandomArray(randnums, nShots);
    }

    // sample bit strings
    CHECK_CUSTATEVECTOR(custatevecSamplerSample(
        handle, sampler, bitStrings_out, bitOrdering, bitStringLen, randnums, nShots,
        CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER));

    // destroy descriptor and handle
    CHECK_CUSTATEVECTOR(custatevecSamplerDestroy(sampler));
    extraWorkspaceSizeInBytes = extraWorkspaceSizeInBytes_CHECK;

    if (freeRandnums)
    {
        delete[] randnums;
        randnums = nullptr;
    }

    return EXIT_SUCCESS;
}

template int sampleSV<precision::bit_32>(custatevecHandle_t &handle,
                                         const int nIndexBits,
                                         const int bitOrdering[], // Qubits to measure
                                         const int bitStringLen,  // length of bitOrdering
                                         PRECISION_TYPE_COMPLEX(precision::bit_32) d_sv[],
                                         custatevecIndex_t bitStrings_out[],
                                         const int nShots,
                                         void *extraWorkspace,
                                         size_t &extraWorkspaceSizeInBytes,
                                         double randnums[]);
template int sampleSV<precision::bit_64>(custatevecHandle_t &handle,
                                         const int nIndexBits,
                                         const int bitOrdering[], // Qubits to measure
                                         const int bitStringLen,  // length of bitOrdering
                                         PRECISION_TYPE_COMPLEX(precision::bit_64) d_sv[],
                                         custatevecIndex_t bitStrings_out[],
                                         const int nShots,
                                         void *extraWorkspace,
                                         size_t &extraWorkspaceSizeInBytes,
                                         double randnums[]);