#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <custatevec.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "../CudaControl/Helper.hpp"

#include "../functionality/randomArray.hpp"
#include "ApplySampler.hpp"

int sampleSV(custatevecHandle_t &handle,
             const int nIndexBits,
             const int bitOrdering[], // Qubits to measure
             const int bitStringLen,  // length of bitOrdering
             cuDoubleComplex d_sv[],
             custatevecIndex_t bitStrings_out[],
             const int nShots,
             void *extraWorkspace,
             size_t &extraWorkspaceSizeInBytes,
             double randnums[])
{
    size_t extraWorkspaceSizeInBytes_CHECK = extraWorkspaceSizeInBytes;
    int nMaxShots = nShots;

    custatevecSamplerDescriptor_t sampler;
    // create sampler and check the size of external workspace
    CHECK_CUSTATEVECTOR(custatevecSamplerCreate(
        handle, d_sv, CUDA_C_64F, nIndexBits, &sampler, nMaxShots,
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

int sampleSV(custatevecHandle_t &handle,
             const int nIndexBits,
             const std::vector<int> &bitOrdering,
             cuDoubleComplex d_sv[],
             custatevecIndex_t bitStrings_out[],
             const int nShots,
             void *extraWorkspace,
             size_t &extraWorkspaceSizeInBytes,
             double randnums[])
{
    CHECK_BROAD_ERROR(
        sampleSV(
            handle,
            nIndexBits,
            bitOrdering.data(),
            bitOrdering.size(),
            d_sv,
            bitStrings_out,
            nShots,
            extraWorkspace,
            extraWorkspaceSizeInBytes,
            randnums));
    return cudaSuccess;
}

int sampleSV(custatevecHandle_t &handle,
             const int nIndexBits,
             const std::vector<int> &bitOrdering,
             cuDoubleComplex d_sv[],
             std::vector<custatevecIndex_t> &bitStrings_out,
             const int nShots,
             void *extraWorkspace,
             size_t &extraWorkspaceSizeInBytes,
             double randnums[])
{
    bitStrings_out.reserve(nShots);
    CHECK_BROAD_ERROR(
        sampleSV(
            handle,
            nIndexBits,
            bitOrdering.data(),
            bitOrdering.size(),
            d_sv,
            bitStrings_out.data(),
            nShots,
            extraWorkspace,
            extraWorkspaceSizeInBytes,
            randnums));
    return cudaSuccess;
}
