#include <cuda_runtime_api.h> 
#include <cuComplex.h>        
#include <custatevec.h>       
#include <stdio.h>            
#include <stdlib.h>           
#include <iostream>
#include <array>
#include <bitset>
#include "Precision.hpp"
#include "../CudaControl/Helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR
#include "ApplyGates.hpp"

template <precision SelectPrecision>
int applyGatesGeneral(custatevecHandle_t &handle,
                      const int nIndexBits,
                      const PRECISION_TYPE_COMPLEX(SelectPrecision) matrix[],
                      const int adjoint,
                      const int targets[],
                      const int nTargets,
                      const int controls[],
                      const int nControls,
                      PRECISION_TYPE_COMPLEX(SelectPrecision) *d_sv,
                      void *extraWorkspace,
                      size_t &extraWorkspaceSizeInBytes)
{
    cudaDataType_t cudaType;
    custatevecComputeType_t custatevecType;
    if constexpr (SelectPrecision == precision::bit_32)
    {
        cudaType = CUDA_C_32F;
        custatevecType = CUSTATEVEC_COMPUTE_32F;
    }
    else if constexpr (SelectPrecision == precision::bit_64)
    {
        cudaType = CUDA_C_64F;
        custatevecType = CUSTATEVEC_COMPUTE_64F;
    }

    size_t extraWorkspaceSizeInBytes_CHECK = extraWorkspaceSizeInBytes;
    // check the size of external workspace
    CHECK_CUSTATEVECTOR(custatevecApplyMatrixGetWorkspaceSize(
        handle, cudaType, nIndexBits, matrix, cudaType, CUSTATEVEC_MATRIX_LAYOUT_ROW,
        adjoint, nTargets, nControls, custatevecType, &extraWorkspaceSizeInBytes_CHECK));

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
        handle, d_sv, cudaType, nIndexBits, matrix, cudaType,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets, nTargets, controls, nullptr,
        nControls, custatevecType, extraWorkspace, extraWorkspaceSizeInBytes_CHECK));

    extraWorkspaceSizeInBytes = extraWorkspaceSizeInBytes_CHECK;

    return cudaSuccess;
}

template int applyGatesGeneral<precision::bit_32>(custatevecHandle_t &handle,
                                                  const int nIndexBits,
                                                  const PRECISION_TYPE_COMPLEX(precision::bit_32) matrix[],
                                                  const int adjoint,
                                                  const int targets[],
                                                  const int nTargets,
                                                  const int controls[],
                                                  const int nControls,
                                                  PRECISION_TYPE_COMPLEX(precision::bit_32) *d_sv,
                                                  void *extraWorkspace,
                                                  size_t &extraWorkspaceSizeInBytes);

template int applyGatesGeneral<precision::bit_64>(custatevecHandle_t &handle,
                                                  const int nIndexBits,
                                                  const PRECISION_TYPE_COMPLEX(precision::bit_64) matrix[],
                                                  const int adjoint,
                                                  const int targets[],
                                                  const int nTargets,
                                                  const int controls[],
                                                  const int nControls,
                                                  PRECISION_TYPE_COMPLEX(precision::bit_64) *d_sv,
                                                  void *extraWorkspace,
                                                  size_t &extraWorkspaceSizeInBytes);

