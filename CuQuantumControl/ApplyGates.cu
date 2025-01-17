#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <array>
#include <bitset>
#include "../CudaControl/Helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR
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
                      size_t &extraWorkspaceSizeInBytes)
{
    size_t extraWorkspaceSizeInBytes_CHECK = extraWorkspaceSizeInBytes;
    // check the size of external workspace
    CHECK_CUSTATEVECTOR(custatevecApplyMatrixGetWorkspaceSize(
        handle, CUDA_C_64F, nIndexBits, matrix, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
        adjoint, nTargets, nControls, CUSTATEVEC_COMPUTE_64F, &extraWorkspaceSizeInBytes_CHECK));

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes_CHECK > extraWorkspaceSizeInBytes)
    {
        std::cout << "Extra space needed: " << extraWorkspaceSizeInBytes_CHECK - extraWorkspaceSizeInBytes << " Bytes\n";
        if (extraWorkspace != nullptr)
        {
            CHECK_CUDA(cudaFree(extraWorkspace));
        }
        CHECK_CUDA(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes_CHECK));
    }
    // apply gate
    CHECK_CUSTATEVECTOR(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_64F, nIndexBits, matrix, CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets, nTargets, controls, nullptr,
        nControls, CUSTATEVEC_COMPUTE_64F, extraWorkspace, extraWorkspaceSizeInBytes_CHECK));

    extraWorkspaceSizeInBytes = extraWorkspaceSizeInBytes_CHECK;

    return cudaSuccess;
}

#define DEFINE_GATE_APPLY_FUNCTION_BACKEND(FUNC_NAME, MATRIX_VALUES) \
    int FUNC_NAME(custatevecHandle_t &handle,                        \
                  const int nIndexBits,                              \
                  const int adjoint,                                 \
                  const int target,                                  \
                  cuDoubleComplex *d_sv,                             \
                  void *extraWorkspace,                              \
                  size_t &extraWorkspaceSizeInBytes)                 \
    {                                                                \
        constexpr cuDoubleComplex matrix[] = MATRIX_VALUES;          \
        CHECK_BROAD_ERROR(applyGatesGeneral(                         \
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
            extraWorkspaceSizeInBytes));                             \
        return CUSTATEVEC_STATUS_SUCCESS;                            \
    }                                                                \
    int FUNC_NAME(custatevecHandle_t &handle,                        \
                  const int nIndexBits,                              \
                  const int adjoint,                                 \
                  const int target,                                  \
                  const int controls[],                              \
                  const int nControls,                               \
                  cuDoubleComplex *d_sv,                             \
                  void *extraWorkspace,                              \
                  size_t &extraWorkspaceSizeInBytes)                 \
    {                                                                \
        constexpr cuDoubleComplex matrix[] = MATRIX_VALUES;          \
        CHECK_BROAD_ERROR(applyGatesGeneral(                         \
            handle,                                                  \
            nIndexBits,                                              \
            matrix,                                                  \
            adjoint,                                                 \
            &target,                                                 \
            1,                                                       \
            controls,                                                \
            nControls,                                               \
            d_sv,                                                    \
            extraWorkspace,                                          \
            extraWorkspaceSizeInBytes));                             \
        return CUSTATEVEC_STATUS_SUCCESS;                            \
    }

DEFINE_GATE_APPLY_FUNCTION_BACKEND(applyH, HMat)
DEFINE_GATE_APPLY_FUNCTION_BACKEND(applyX, XMat)
DEFINE_GATE_APPLY_FUNCTION_BACKEND(applyY, YMat)
DEFINE_GATE_APPLY_FUNCTION_BACKEND(applyZ, ZMat)
