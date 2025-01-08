#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <bitset>
#include "helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR
#include "ApplyGates.hpp"

int applyGatesGeneral(custatevecHandle_t &handle,
                      const int nIndexBits,
                      const cuDoubleComplex matrix[],
                      const int adjoint,
                      const int targets[],
                      const int nTargets,
                      const int controls[],
                      const int nControls,
                      cuDoubleComplex *d_sv,
                      void *extraWorkspace,
                      size_t extraWorkspaceSizeInBytes)
{

    // check the size of external workspace
    HANDLE_ERROR(custatevecApplyMatrixGetWorkspaceSize(
        handle, CUDA_C_64F, nIndexBits, matrix, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
        adjoint, nTargets, nControls, CUSTATEVEC_COMPUTE_64F, &extraWorkspaceSizeInBytes));

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
    {
        std::cout << "Extra space needed: " << extraWorkspaceSizeInBytes << " Bytes";
        if (extraWorkspace != nullptr)
        {
            HANDLE_CUDA_ERROR(cudaFree(extraWorkspace));
        }
        HANDLE_CUDA_ERROR(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
    }
    // apply gate
    HANDLE_ERROR(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_64F, nIndexBits, matrix, CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets, nTargets, controls, nullptr,
        nControls, CUSTATEVEC_COMPUTE_64F, extraWorkspace, extraWorkspaceSizeInBytes));
    if (extraWorkspace != nullptr)
    {
        HANDLE_CUDA_ERROR(cudaFree(extraWorkspace));
    }
    return cudaSuccess;
}

#define DEFINE_GATE_APPLY_FUNCTION_BACKEND(FUNC_NAME, MATRIX_VALUES) \
    int FUNC_NAME(custatevecHandle_t &handle,                        \
                  const int nIndexBits,                              \
                  const int adjoint,                                 \
                  const int target,                                  \
                  cuDoubleComplex *d_sv,                             \
                  void *extraWorkspace,                              \
                  size_t extraWorkspaceSizeInBytes)                  \
    {                                                                \
        cuDoubleComplex matrix[] = MATRIX_VALUES;                    \
        return applyGatesGeneral(                                           \
            handle,                                                  \
            nIndexBits,                                              \
            matrix,                                                  \
            adjoint,                                                 \
            &target,                                                 \
            1,                                                       \
            {},                                                      \
            0,                                                       \
            d_sv,                                                    \
            extraWorkspace,                                          \
            extraWorkspaceSizeInBytes);                              \
    }

DEFINE_GATE_APPLY_FUNCTION_BACKEND(applyH, HMat)
DEFINE_GATE_APPLY_FUNCTION_BACKEND(applyX, XMat)
DEFINE_GATE_APPLY_FUNCTION_BACKEND(applyY, YMat)
DEFINE_GATE_APPLY_FUNCTION_BACKEND(applyZ, ZMat)
