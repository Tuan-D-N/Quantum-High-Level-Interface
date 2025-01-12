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

        custatevecHandle_t handle = NULL;
        CHECK_CUSTATEVECTOR(custatevecCreate(&handle));
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        custatevecIndex_t outBitString[nShots];
        sampleSV<nIndexBits>(handle, nIndexBits, {0, 1, 2}, d_sv, outBitString, nShots, extraWorkspace, extraWorkspaceSizeInBytes);
        
        for(int i = 0; i < nShots; ++i)
        {
            std::cout << outBitString[i];
        }
    }

    return cudaSuccess;
}

int grover()
{

    const int nIndexBits = 3;
    const int nSvSize = (1 << nIndexBits);
    const int adjoint = 0;
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
        constexpr auto allQubit = range(0, nIndexBits);
        constexpr auto allQubitExceptLast = range(0, nIndexBits - 1);

        CHECK_BROAD_ERROR(applyH<nIndexBits>(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));

        for (int i = 0; i < 10; ++i)
        {
            // Mark
            constexpr int markTarget = nIndexBits - 1; // lastQubit
            CHECK_BROAD_ERROR(applyZ<nIndexBits - 1>(handle, nIndexBits, adjoint, markTarget, allQubitExceptLast, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));

            // Diffusion
            CHECK_BROAD_ERROR(applyH<nIndexBits>(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
            CHECK_BROAD_ERROR(applyX<nIndexBits>(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
            CHECK_BROAD_ERROR(applyZ<nIndexBits - 1>(handle, nIndexBits, adjoint, markTarget, allQubitExceptLast, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
            CHECK_BROAD_ERROR(applyX<nIndexBits>(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
            CHECK_BROAD_ERROR(applyH<nIndexBits>(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
        }

        // Algo ------------------------------------------------------------
        CHECK_BROAD_ERROR(custatevecDestroy(handle));
        if (extraWorkspace != nullptr)
            CHECK_CUDA(cudaFree(extraWorkspace));

        // Grover ----------------------------------------------------------------------------------------
        CHECK_CUDA(cudaFree(d_sv));
    }

    return cudaSuccess;
}