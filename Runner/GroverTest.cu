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

int grover(const int nIndexBits)
{
    const int nSvSize = (1 << nIndexBits);
    const int adjoint = 0;
    const int nShots = 100;
    {
        auto timer = Timer("Grover Cuquantum C++ qubits = " + std::to_string(nIndexBits));

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

int grover2(const int nIndexBits)
{
    const int nSvSize = (1 << nIndexBits);
    const int adjoint = 0;
    const int nShots = 100;
    {
        auto timer = Timer("Grover Cuquantum C++ qubits = " + std::to_string(nIndexBits));

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

        (applyH(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));

        for (int i = 0; i < 10; ++i)
        {
            // Mark
            int markTarget = nIndexBits - 1; // lastQubit
            (applyZ(handle, nIndexBits, adjoint, markTarget, allQubitExceptLast, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));

            // Diffusion
            (applyH(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
            (applyX(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
            (applyZ(handle, nIndexBits, adjoint, markTarget, allQubitExceptLast, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
            (applyX(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
            (applyH(handle, nIndexBits, adjoint, allQubit, d_sv, extraWorkspace, extraWorkspaceSizeInBytes));
        }
        std::vector<custatevecIndex_t> outBitString;
        (sampleSV(handle, nIndexBits, allQubit, d_sv, outBitString, nShots, extraWorkspace, extraWorkspaceSizeInBytes));

        // Algo ------------------------------------------------------------
        CHECK_CUSTATEVECTOR(custatevecDestroy(handle));
        if (extraWorkspace != nullptr)
            CHECK_CUDA(cudaFree(extraWorkspace));

        // Grover ----------------------------------------------------------------------------------------
        CHECK_CUDA(cudaFree(d_sv));
    }

    return cudaSuccess;
}

int grover3(const int nIndexBits)
{
    const int svSize = (1 << nIndexBits);
    cuDoubleComplex xMat[] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
    cuDoubleComplex zMat[] = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}};
    cuDoubleComplex hMat[] = {{INV_SQRT2, 0.0}, {INV_SQRT2, 0.0}, {INV_SQRT2, 0.0}, {-INV_SQRT2, 0.0}};

    cuDoubleComplex *d_sv;
    CHECK_CUDA(cudaMallocManaged((void **)&d_sv, svSize * sizeof(cuDoubleComplex)));

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
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

    for (int i = 0; i < nIndexBits; ++i)
    {
        custatevecApplyMatrix(
            handle, d_sv, CUDA_C_64F, nIndexBits, hMat, CUDA_C_64F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, (int[]){i}, 1, {}, nullptr,
            0, CUSTATEVEC_COMPUTE_64F, extraWorkspace, extraWorkspaceSizeInBytes);
    }

    // destroy handle
    CHECK_CUSTATEVECTOR(custatevecDestroy(handle));

    //----------------------------------------------------------------------------------------------

    printDeviceArray(d_sv, svSize);
    CHECK_CUDA(cudaFree(d_sv));
    return EXIT_SUCCESS;
}