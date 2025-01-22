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

#include "../CudaControl/Helper.hpp"
#include <iostream>

int main(void)
{

    const int nIndexBits = 3;
    const int nSvSize = (1 << nIndexBits);

    const int bitOrderingLen = 2;
    const int bitOrdering[] = {1,2}; //Order of fastest changing qubit down the state vector.

    const int maskLen = 1;
    const int maskBitString[] = {1}; //qubit value at that index {0,1}
    const int maskOrdering[] = {0}; //qubit index number

    const int bufferSize = -5;
    const int accessBegin = 0;
    const int accessEnd = 4;

    cuDoubleComplex h_sv[] = {{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1}, {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3}, {0.3, 0.4}, {0.4, 0.5}};
    cuDoubleComplex buffer[] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    cuDoubleComplex buffer_result[] = {{0.3, 0.3}, {0.1, 0.2}, {0.4, 0.5}};

    custatevecAccessorDescriptor_t accessor;

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

    // create accessor and check the size of external workspace
    CHECK_CUSTATEVECTOR(custatevecAccessorCreateView(
        handle, d_sv, CUDA_C_64F, nIndexBits, &accessor, bitOrdering, bitOrderingLen,
        maskBitString, maskOrdering, maskLen, &extraWorkspaceSizeInBytes));

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
        CHECK_CUDA(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));

    // set external workspace
    CHECK_CUSTATEVECTOR(custatevecAccessorSetExtraWorkspace(
        handle, accessor, extraWorkspace, extraWorkspaceSizeInBytes));

    // get state vector components
    CHECK_CUSTATEVECTOR(custatevecAccessorGet(
        handle, accessor, buffer, accessBegin, accessEnd));

    // destroy descriptor and handle
    CHECK_CUSTATEVECTOR(custatevecAccessorDestroy(accessor));
    CHECK_CUSTATEVECTOR(custatevecDestroy(handle));

    //----------------------------------------------------------------------------------------------

    CHECK_CUDA(cudaMemcpy(h_sv, d_sv, nSvSize * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; ++i)
    {
        std::cout << buffer[i].x << " , " << buffer[i].y << "\n";
    }

    return EXIT_SUCCESS;
}