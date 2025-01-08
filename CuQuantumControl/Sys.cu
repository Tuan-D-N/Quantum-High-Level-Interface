/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <bitset>
#include "helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR
#include "ApplyGates.hpp"
#include <cstring>
int runner()
{

    const int nIndexBits = 3;
    const int nSvSize = (1 << nIndexBits);
    const int adjoint = 0;


    cuDoubleComplex matrix[] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
    cuDoubleComplex h_sv[] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    cuDoubleComplex *d_sv;
    HANDLE_CUDA_ERROR(cudaMallocManaged((void **)&d_sv, nSvSize * sizeof(cuDoubleComplex)));

    // Initialize the values
    std::memcpy(d_sv, &h_sv, nSvSize * sizeof(cuDoubleComplex));

    custatevecHandle_t handle;
    HANDLE_ERROR(custatevecCreate(&handle));
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    applyX(handle, nIndexBits, (int)false, adjoint, d_sv, extraWorkspace, extraWorkspaceSizeInBytes);



    for (int i = 0; i < nSvSize; i++)
    {
        std::cout << (d_sv[i].x) << "," << d_sv[i].y << " , " << static_cast<std::bitset<3>>(i) << std::endl;
    }

    HANDLE_CUDA_ERROR(cudaFree(d_sv));

    return cudaSuccess;
}