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

int runner()
{

    const int nIndexBits = 3;
    const int nSvSize = (1 << nIndexBits);
    const int nTargets = 1;
    const int nControls = 2;
    const int adjoint = 0;

    const int targets[] = {2};
    const int controls[] = {0, 1};

    cuDoubleComplex matrix[] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

    cuDoubleComplex *d_sv;
    HANDLE_CUDA_ERROR(cudaMallocManaged((void **)&d_sv, nSvSize * sizeof(cuDoubleComplex)));

    // Initialize the values
    d_sv[0] = make_cuDoubleComplex(1.0, 0.0);
    d_sv[1] = make_cuDoubleComplex(2.0, 0.0);
    d_sv[2] = make_cuDoubleComplex(3.0, 0.0);
    d_sv[3] = make_cuDoubleComplex(4.0, 0.0);
    d_sv[4] = make_cuDoubleComplex(5.0, 0.0);
    d_sv[5] = make_cuDoubleComplex(6.0, 0.0);
    d_sv[6] = make_cuDoubleComplex(7.0, 0.0);
    d_sv[7] = make_cuDoubleComplex(8.0, 0.0);

    custatevecHandle_t handle;
    HANDLE_ERROR(custatevecCreate(&handle));
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    applyX(handle, nIndexBits, (int)false, 0, d_sv, extraWorkspace, extraWorkspaceSizeInBytes);

    for (int i = 0; i < nSvSize; i++)
    {
        std::cout << (d_sv[i].x) << "," << d_sv[i].y << " , " << static_cast<std::bitset<3>>(i) << std::endl;
    }

    HANDLE_CUDA_ERROR(cudaFree(d_sv));
}