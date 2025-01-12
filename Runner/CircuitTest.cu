#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <bitset>
#include "../CudaControl/Helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR
#include "../CuQuantumControl/ApplyGates.hpp"
#include "../CuQuantumControl/ApplySampler.hpp"
#include "../CuQuantumControl/QftStateVec.hpp"
#include "../CuQuantumControl/Utilities.hpp"
#include "../functionality/fftShift.hpp"
#include "../functionality/ClockTimer.hpp"
#include "../functionality/RangeCompileTime.hpp"
#include "../functionality/Utilities.hpp"
#include <cstring>

int runner1()
{
    const int nIndexBits = 3;
    const int nSvSize = (1 << nIndexBits);
    const int adjoint = false;
    const int nShots = 5;

    {
        cuDoubleComplex *d_sv;
        CHECK_CUDA(cudaMallocManaged((void **)&d_sv, nSvSize * sizeof(cuDoubleComplex)));
        set2NoState(d_sv, nSvSize);
        d_sv[0] = {1, 0};

        custatevecHandle_t handle = NULL;
        CHECK_CUSTATEVECTOR(custatevecCreate(&handle));
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        custatevecIndex_t outBitString[nShots];
        custatevecIndex_t bitStrings_result[] = {0b00, 0b01, 0b10, 0b11, 0b11};
        CHECK_BROAD_ERROR(applyX(handle, nIndexBits, adjoint, std::vector<int>{0, 1, 2}, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
        CHECK_BROAD_ERROR(sampleSV(handle, nIndexBits, {0, 1, 2}, d_sv, outBitString, nShots, extraWorkspace, extraWorkspaceSizeInBytes));
        printDeviceArray(d_sv, nSvSize);

        for (int i = 0; i < nShots; ++i)
        {
            std::cout << std::bitset<nSvSize>(outBitString[i]) << " , " << bitStrings_result[i] << "\n";
        }

        if (extraWorkspace != nullptr)
            CHECK_CUDA(cudaFree(extraWorkspace));
    }

    return cudaSuccess;
}

int grover(const int nIndexBits)
{
    const int nSvSize = (1 << nIndexBits);
    const int adjoint = 0;
    const int nShots = 100;
    {
        Timer("Grover Cuquantum C++ qubits = " + std::to_string(nIndexBits));

        // Make the statevector -------------------------------------------------------------------------------
        cuDoubleComplex *d_sv;
        CHECK_CUDA(cudaMallocManaged((void **)&d_sv, nSvSize * sizeof(cuDoubleComplex)));
        d_sv[0] = {1, 0};
        for (int i = 1; i < nSvSize; ++i)
        {
            d_sv[i] = {0, 0};
        }
        // Make the statevector -------------------------------------------------------------------------------

        // Grover ----------------------------------------------------------------------------------------
        custatevecHandle_t handle;
        CHECK_CUSTATEVECTOR(custatevecCreate(&handle));
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        // Algo ------------------------------------------------------------
        std::vector<int> allQubit = rangeVec(0, nIndexBits);
        std::vector<int> allQubitExceptLast = rangeVec(0, nIndexBits - 1);

        CHECK_BROAD_ERROR(applyH(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));

        for (int i = 0; i < 10; ++i)
        {
            // Mark
            int markTarget = nIndexBits - 1; // lastQubit
            CHECK_BROAD_ERROR(applyZ(handle, nIndexBits, adjoint, markTarget, allQubitExceptLast, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));

            // Diffusion
            CHECK_BROAD_ERROR(applyH(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
            CHECK_BROAD_ERROR(applyX(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
            CHECK_BROAD_ERROR(applyZ(handle, nIndexBits, adjoint, markTarget, allQubitExceptLast, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
            CHECK_BROAD_ERROR(applyX(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
            CHECK_BROAD_ERROR(applyH(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
        }
        std::vector<custatevecIndex_t> outBitString;
        CHECK_BROAD_ERROR(sampleSV(handle, nIndexBits, allQubit, d_sv, outBitString, nShots, extraWorkspace, extraWorkspaceSizeInBytes));

        // Algo ------------------------------------------------------------
        CHECK_BROAD_ERROR(custatevecDestroy(handle));
        if (extraWorkspace != nullptr)
            CHECK_CUDA(cudaFree(extraWorkspace));

        // Grover ----------------------------------------------------------------------------------------
        CHECK_CUDA(cudaFree(d_sv));
    }

    return cudaSuccess;
}

int runner2(void)
{

    const int nIndexBits = 3;
    const int nSvSize = (1 << nIndexBits);
    const int nMaxShots = 5;
    const int nShots = 5;

    const int bitStringLen = 2;
    const int bitOrdering[] = {0, 1};

    custatevecIndex_t bitStrings[nShots];
    custatevecIndex_t bitStrings_result[] = {0b00, 0b01, 0b10, 0b11, 0b11};

    cuDoubleComplex h_sv[] = {{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3}, {0.3, 0.4}, {0.4, 0.5}};

    // In real appliction, random numbers in range [0, 1) will be used.
    const double randnums[] = {0.1, 0.8, 0.4, 0.6, 0.2};

    custatevecSamplerDescriptor_t sampler;

    cuDoubleComplex *d_sv;
    CHECK_CUDA(cudaMalloc((void **)&d_sv, nSvSize * sizeof(cuDoubleComplex)));

    CHECK_CUDA(cudaMemcpy(d_sv, h_sv, nSvSize * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice));

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    CHECK_CUSTATEVECTOR(custatevecCreate(&handle));

    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    // create sampler and check the size of external workspace
    CHECK_CUSTATEVECTOR(custatevecSamplerCreate(
        handle, d_sv, CUDA_C_64F, nIndexBits, &sampler, nMaxShots,
        &extraWorkspaceSizeInBytes));

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
        CHECK_CUDA(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));

    // sample preprocess
    CHECK_CUSTATEVECTOR(custatevecSamplerPreprocess(
        handle, sampler, extraWorkspace, extraWorkspaceSizeInBytes));

    // sample bit strings
    CHECK_CUSTATEVECTOR(custatevecSamplerSample(
        handle, sampler, bitStrings, bitOrdering, bitStringLen, randnums, nShots,
        CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER));

    // destroy descriptor and handle
    CHECK_CUSTATEVECTOR(custatevecSamplerDestroy(sampler));
    CHECK_CUSTATEVECTOR(custatevecDestroy(handle));

    //----------------------------------------------------------------------------------------------

    CHECK_CUDA(cudaMemcpy(h_sv, d_sv, nSvSize * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < nShots; i++)
    {
        if (bitStrings[i] != bitStrings_result[i])
        {
            correct = false;
            break;
        }
    }

    CHECK_CUDA(cudaFree(d_sv));
    if (extraWorkspaceSizeInBytes)
        CHECK_CUDA(cudaFree(extraWorkspace));

    for (int i = 0; i < nShots; ++i)
    {
        std::cout << bitStrings[i] << " , " << bitStrings_result[i] << "\n";
    }

    return cudaSuccess;
}

int runner3(void)
{

    const int nIndexBits = 3;
    const int nSvSize = (1 << nIndexBits);
    const int nMaxShots = 5;
    const int nShots = 5;

    const int bitStringLen = 2;
    const int bitOrdering[] = {0, 1};

    custatevecIndex_t bitStrings[nShots];
    custatevecIndex_t bitStrings_result[] = {0b00, 0b01, 0b10, 0b11, 0b11};

    cuDoubleComplex h_sv[] = {{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3}, {0.3, 0.4}, {0.4, 0.5}};

    // In real appliction, random numbers in range [0, 1) will be used.
    const double randnums[] = {0.1, 0.8, 0.4, 0.6, 0.2};

    custatevecSamplerDescriptor_t sampler;

    cuDoubleComplex *d_sv;
    CHECK_CUDA(cudaMalloc((void **)&d_sv, nSvSize * sizeof(cuDoubleComplex)));

    CHECK_CUDA(cudaMemcpy(d_sv, h_sv, nSvSize * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice));

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    CHECK_CUSTATEVECTOR(custatevecCreate(&handle));

    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    CHECK_BROAD_ERROR(sampleSV<nIndexBits>(handle, nIndexBits, {0, 1, 2}, d_sv, bitStrings, nShots, extraWorkspace, extraWorkspaceSizeInBytes));
    CHECK_CUSTATEVECTOR(custatevecDestroy(handle));

    //----------------------------------------------------------------------------------------------

    CHECK_CUDA(cudaMemcpy(h_sv, d_sv, nSvSize * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < nShots; i++)
    {
        if (bitStrings[i] != bitStrings_result[i])
        {
            correct = false;
            break;
        }
    }

    CHECK_CUDA(cudaFree(d_sv));
    if (extraWorkspaceSizeInBytes)
        CHECK_CUDA(cudaFree(extraWorkspace));

    for (int i = 0; i < nShots; ++i)
    {
        std::cout << bitStrings[i] << " , " << bitStrings_result[i] << "\n";
    }

    return cudaSuccess;
}