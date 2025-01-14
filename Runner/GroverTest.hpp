#pragma once 
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
#include "../functionality/randomArray.hpp"
#include <cstring>

int grover(const int nIndexBits);
int grover1(const int nIndexBits);
int grover2(const int nIndexBits);
template <int nIndexBits>
int grover3()
{
    constexpr int svSize = (1 << nIndexBits);

    const int nShots = 100;
    const int nMaxShots = nShots;
    int bitOrdering[nIndexBits] = {};
    for (int i = 0; i < nIndexBits; ++i)
    {
        bitOrdering[i] = i;
    }
    const int bitStringLen = nIndexBits;
    custatevecIndex_t bitStrings[nShots];
    double randnums[nShots] = {};
    generateRandomArray(randnums, nShots);

    cuDoubleComplex xMat[] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
    cuDoubleComplex zMat[] = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}};
    cuDoubleComplex hMat[] = {{INV_SQRT2, 0.0}, {INV_SQRT2, 0.0}, {INV_SQRT2, 0.0}, {-INV_SQRT2, 0.0}};

    cuDoubleComplex *d_sv;
    CHECK_CUDA(cudaMallocManaged((void **)&d_sv, svSize * sizeof(cuDoubleComplex)));

    //----------------------------------------------------------------------------------------------

    {
        auto timer = Timer("Grover Cuquantum C++ qubits = " + std::to_string(nIndexBits));

        int controlsAll[nIndexBits];
        int controlsAllExceptLast[nIndexBits - 1];
        int markTargets[] = {nIndexBits - 1};
        for (int i = 0; i < nIndexBits - 1; ++i)
        {
            controlsAll[i] = i;
            controlsAllExceptLast[i] = i;
        }
        controlsAll[nIndexBits - 1] = nIndexBits - 1;

        // custatevec handle initialization
        custatevecSamplerDescriptor_t sampler;
        custatevecHandle_t handle;
        CHECK_CUSTATEVECTOR(custatevecCreate(&handle));
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        // Init to zero state
        d_sv[0] = {1, 0};
        for (int i = 1; i < svSize; ++i)
        {
            d_sv[i] = {0, 0};
        }
        // H to all qubits
        for (int i = 0; i < nIndexBits; ++i)
        {
            int targets[] = {i};
            CHECK_CUSTATEVECTOR(custatevecApplyMatrix(
                handle, d_sv, CUDA_C_64F, nIndexBits, hMat, CUDA_C_64F,
                CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, {}, nullptr,
                0, CUSTATEVEC_COMPUTE_64F, extraWorkspace, extraWorkspaceSizeInBytes));
        }
        // H to all qubits

        for (int i = 0; i < 10; ++i)
        {
            // mark
            CHECK_CUSTATEVECTOR(custatevecApplyMatrix(
                handle, d_sv, CUDA_C_64F, nIndexBits, zMat, CUDA_C_64F,
                CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, markTargets, 1, controlsAllExceptLast, nullptr,
                nIndexBits - 1, CUSTATEVEC_COMPUTE_64F, extraWorkspace, extraWorkspaceSizeInBytes));
            // Diffusion
            // H->all, X->all, cz->allexceptLast mark, x->all, H->all
            for (int j = 0; j < nIndexBits; ++j)
            {
                int targets[] = {j};
                CHECK_CUSTATEVECTOR(custatevecApplyMatrix(
                    handle, d_sv, CUDA_C_64F, nIndexBits, hMat, CUDA_C_64F,
                    CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, {}, nullptr,
                    0, CUSTATEVEC_COMPUTE_64F, extraWorkspace, extraWorkspaceSizeInBytes));
            }
            for (int j = 0; j < nIndexBits; ++j)
            {
                int targets[] = {j};
                CHECK_CUSTATEVECTOR(custatevecApplyMatrix(
                    handle, d_sv, CUDA_C_64F, nIndexBits, xMat, CUDA_C_64F,
                    CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, {}, nullptr,
                    0, CUSTATEVEC_COMPUTE_64F, extraWorkspace, extraWorkspaceSizeInBytes));
            }
            CHECK_CUSTATEVECTOR(custatevecApplyMatrix(
                handle, d_sv, CUDA_C_64F, nIndexBits, zMat, CUDA_C_64F,
                CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, markTargets, 1, controlsAllExceptLast, nullptr,
                nIndexBits - 1, CUSTATEVEC_COMPUTE_64F, extraWorkspace, extraWorkspaceSizeInBytes));
            for (int j = 0; j < nIndexBits; ++j)
            {
                int targets[] = {j};
                CHECK_CUSTATEVECTOR(custatevecApplyMatrix(
                    handle, d_sv, CUDA_C_64F, nIndexBits, xMat, CUDA_C_64F,
                    CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, {}, nullptr,
                    0, CUSTATEVEC_COMPUTE_64F, extraWorkspace, extraWorkspaceSizeInBytes));
            }
            for (int j = 0; j < nIndexBits; ++j)
            {
                int targets[] = {j};
                CHECK_CUSTATEVECTOR(custatevecApplyMatrix(
                    handle, d_sv, CUDA_C_64F, nIndexBits, hMat, CUDA_C_64F,
                    CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, {}, nullptr,
                    0, CUSTATEVEC_COMPUTE_64F, extraWorkspace, extraWorkspaceSizeInBytes));
            }
        }

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


        // std::cout << nShots << "\n";
        // std::cout << bitStringLen << "\n";
        
        //     for(int k = 0; k < nShots; ++k)
        //     {
        //         // std::cout << randnums[k] << "\n";
        //         // std::cout << bitStrings[k] << "\n";

        //     }
        //     for(int k = 0; k < nIndexBits; ++k)
        //     {
        //         std::cout << bitOrdering[k] << "\n";

        //     }

        // sample bit strings
        CHECK_CUSTATEVECTOR(custatevecSamplerSample(
            handle, sampler, bitStrings, bitOrdering, bitStringLen, randnums, nShots,
            CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER));

        // destroy descriptor and handle
        CHECK_CUSTATEVECTOR(custatevecSamplerDestroy(sampler));

        //  destroy handle
        CHECK_CUSTATEVECTOR(custatevecDestroy(handle));
        if (extraWorkspaceSizeInBytes)
            CHECK_CUDA(cudaFree(extraWorkspace));
    }
    //----------------------------------------------------------------------------------------------

    // printDeviceArray(d_sv, svSize);
    CHECK_CUDA(cudaFree(d_sv));

    return EXIT_SUCCESS;
}
int grover4(const int nIndexBits);