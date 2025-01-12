#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <bitset>
#include "../CudaControl/Helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR
#include "ApplyGates.hpp"
#include "ApplySampler.hpp"
#include "QftStateVec.hpp"
#include "../functionality/fftShift.hpp"
#include "../functionality/ClockTimer.hpp"
#include <cstring>

int runner()
{

    const int nIndexBits = 3;
    const int nSvSize = (1 << nIndexBits);
    const int adjoint = 0;

    cuDoubleComplex h_sv[] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    cuDoubleComplex *d_sv;
    CHECK_CUDA(cudaMallocManaged((void **)&d_sv, nSvSize * sizeof(cuDoubleComplex)));

    // Initialize the values
    std::memcpy(d_sv, &h_sv, nSvSize * sizeof(cuDoubleComplex));

    for (int i = 0; i < nSvSize; i++)
    {
        std::cout << (d_sv[i].x) << "," << d_sv[i].y << " , " << static_cast<std::bitset<3>>(i) << std::endl;
    }
    std::cout << "\n";

    fftshift1D(d_sv, nSvSize);
    CHECK_CUDA(static_cast<cudaError_t>(ApplyQFTOnStateVector(d_sv, nIndexBits)));
    fftshift1D(d_sv, nSvSize);

    for (int i = 0; i < nSvSize; i++)
    {
        std::cout << (d_sv[i].x) << "," << d_sv[i].y << " , " << static_cast<std::bitset<3>>(i) << std::endl;
    }
    std::cout << "\n";

    CHECK_CUDA(cudaFree(d_sv));

    return cudaSuccess;
}
int runner2()
{

    const int nIndexBits = 3;
    const int nSvSize = (1 << nIndexBits);
    const int adjoint = 0;

    cuDoubleComplex h_sv[] = {{1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
    cuDoubleComplex *d_sv;
    CHECK_CUDA(cudaMallocManaged((void **)&d_sv, nSvSize * sizeof(cuDoubleComplex)));

    // Initialize the values
    std::memcpy(d_sv, &h_sv, nSvSize * sizeof(cuDoubleComplex));

    for (int i = 0; i < nSvSize; i++)
    {
        std::cout << (d_sv[i].x) << "," << d_sv[i].y << " , " << static_cast<std::bitset<3>>(i) << std::endl;
    }
    std::cout << "\n";

    custatevecHandle_t handle;
    CHECK_CUSTATEVECTOR(custatevecCreate(&handle));
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    applyX<3>(handle, nIndexBits, adjoint, {0, 1, 2}, d_sv, extraWorkspace, extraWorkspaceSizeInBytes);
    // cuDoubleComplex matrix[] = XMat;
    // const int target[] = {0};
    // const int control[] = {};
    // applyGatesGeneral(handle, nIndexBits, matrix, adjoint, target, 1, control, 0, d_sv, extraWorkspace, extraWorkspaceSizeInBytes);
    // applyX(handle, nIndexBits, adjoint, target[0], control, sizeof(control) / sizeof(control[0]), d_sv, extraWorkspace, extraWorkspaceSizeInBytes);

    CHECK_CUSTATEVECTOR(custatevecDestroy(handle));

    for (int i = 0; i < nSvSize; i++)
    {
        std::cout << (d_sv[i].x) << "," << d_sv[i].y << " , " << static_cast<std::bitset<3>>(i) << std::endl;
    }
    std::cout << "\n";
    CHECK_CUDA(cudaFree(d_sv));

    return cudaSuccess;
}

template <int N>
struct rangeArray
{
    constexpr rangeArray() : arr()
    {
        for (auto i = 0; i != N; ++i)
            arr[i] = i;
    }
    std::array<int, N> arr;
};

int runner3()
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
        constexpr auto allQubit = rangeArray<nIndexBits>().arr;
        constexpr auto allQubitExceptLast = rangeArray<nIndexBits - 1>().arr;

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