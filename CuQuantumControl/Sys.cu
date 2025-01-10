#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <bitset>
#include "helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR
#include "ApplyGates.hpp"
#include "QftStateVec.hpp"
#include "../functionality/fftShift.hpp"
#include <cstring>
int runner()
{

    const int nIndexBits = 3;
    const int nSvSize = (1 << nIndexBits);
    const int adjoint = 0;

    cuDoubleComplex h_sv[] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    cuDoubleComplex *d_sv;
    HANDLE_CUDA_ERROR(cudaMallocManaged((void **)&d_sv, nSvSize * sizeof(cuDoubleComplex)));

    // Initialize the values
    std::memcpy(d_sv, &h_sv, nSvSize * sizeof(cuDoubleComplex));

    for (int i = 0; i < nSvSize; i++)
    {
        std::cout << (d_sv[i].x) << "," << d_sv[i].y << " , " << static_cast<std::bitset<3>>(i) << std::endl;
    }
    std::cout << "\n";

    fftshift1D(d_sv, nSvSize);
    HANDLE_CUDA_ERROR(static_cast<cudaError_t>(ApplyQFTOnStateVector(d_sv, nIndexBits)));
    fftshift1D(d_sv, nSvSize);

    for (int i = 0; i < nSvSize; i++)
    {
        std::cout << (d_sv[i].x) << "," << d_sv[i].y << " , " << static_cast<std::bitset<3>>(i) << std::endl;
    }
    std::cout << "\n";

    HANDLE_CUDA_ERROR(cudaFree(d_sv));

    return cudaSuccess;
}
int runner2()
{

    const int nIndexBits = 3;
    const int nSvSize = (1 << nIndexBits);
    const int adjoint = 0;

    cuDoubleComplex h_sv[] = {{1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
    cuDoubleComplex *d_sv;
    HANDLE_CUDA_ERROR(cudaMallocManaged((void **)&d_sv, nSvSize * sizeof(cuDoubleComplex)));

    // Initialize the values
    std::memcpy(d_sv, &h_sv, nSvSize * sizeof(cuDoubleComplex));

    for (int i = 0; i < nSvSize; i++)
    {
        std::cout << (d_sv[i].x) << "," << d_sv[i].y << " , " << static_cast<std::bitset<3>>(i) << std::endl;
    }
    std::cout << "\n";

    custatevecHandle_t handle;
    HANDLE_ERROR(custatevecCreate(&handle));
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    applyX<3>(handle, nIndexBits, adjoint, {0, 1, 2}, d_sv, extraWorkspace, extraWorkspaceSizeInBytes);
    // cuDoubleComplex matrix[] = XMat;
    // const int target[] = {0};
    // const int control[] = {};
    // applyGatesGeneral(handle, nIndexBits, matrix, adjoint, target, 1, control, 0, d_sv, extraWorkspace, extraWorkspaceSizeInBytes);
    // applyX(handle, nIndexBits, adjoint, target[0], control, sizeof(control) / sizeof(control[0]), d_sv, extraWorkspace, extraWorkspaceSizeInBytes);

    HANDLE_ERROR(custatevecDestroy(handle));

    for (int i = 0; i < nSvSize; i++)
    {
        std::cout << (d_sv[i].x) << "," << d_sv[i].y << " , " << static_cast<std::bitset<3>>(i) << std::endl;
    }
    std::cout << "\n";
    HANDLE_CUDA_ERROR(cudaFree(d_sv));

    return cudaSuccess;
}