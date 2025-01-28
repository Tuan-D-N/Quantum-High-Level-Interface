#pragma once
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <custatevec.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <bitset>
#include "Precision.hpp"
#include "../CudaControl/Helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR
#include "MatriceDefinitions.hpp"
#include <span>

template <precision selectPrecision>
int applyGatesGeneral(custatevecHandle_t &handle,
                      const int nIndexBits,
                      const PRECISION_TYPE_COMPLEX(selectPrecision) matrix[],
                      const int adjoint,
                      const int targets[],
                      const int nTargets,
                      const int controls[],
                      const int nControls,
                      PRECISION_TYPE_COMPLEX(selectPrecision) * d_sv,
                      void *extraWorkspace,
                      size_t &extraWorkspaceSizeInBytes);

template <precision selectPrecision>
int applyGatesGeneral(custatevecHandle_t &handle,
                      const int nIndexBits,
                      const std::span<const PRECISION_TYPE_COMPLEX(selectPrecision)> matrix,
                      const int adjoint,
                      const std::span<const int> targets,
                      const std::span<const int> controls,
                      PRECISION_TYPE_COMPLEX(selectPrecision) * d_sv,
                      void *extraWorkspace,
                      size_t &extraWorkspaceSizeInBytes)
{
    CHECK_BROAD_ERROR(applyGatesGeneral<selectPrecision>(
        handle,
        nIndexBits,
        matrix.data(),
        adjoint,
        targets.data(),
        targets.size(),
        controls.data(),
        controls.size(),
        d_sv,
        extraWorkspace,
        extraWorkspaceSizeInBytes));

    return cudaSuccess;
}

#define DEFINE_GATE_APPLY_FUNCTION(FUNC_NAME, MATRIX_VALUES)                        \
    template <precision selectPrecision = precision::bit_64>                        \
    int FUNC_NAME(custatevecHandle_t &handle,                                       \
                  const int nIndexBits,                                             \
                  const int adjoint,                                                \
                  const int target,                                                 \
                  PRECISION_TYPE_COMPLEX(selectPrecision) * d_sv,                   \
                  void *extraWorkspace,                                             \
                  size_t &extraWorkspaceSizeInBytes)                                \
    {                                                                               \
        constexpr PRECISION_TYPE_COMPLEX(selectPrecision) matrix[] = MATRIX_VALUES; \
        CHECK_BROAD_ERROR(applyGatesGeneral<selectPrecision>(                       \
            handle,                                                                 \
            nIndexBits,                                                             \
            matrix,                                                                 \
            adjoint,                                                                \
            &target,                                                                \
            1,                                                                      \
            {},                                                                     \
            0,                                                                      \
            d_sv,                                                                   \
            extraWorkspace,                                                         \
            extraWorkspaceSizeInBytes));                                            \
        return CUSTATEVEC_STATUS_SUCCESS;                                           \
    }                                                                               \
    template <precision selectPrecision = precision::bit_64>                        \
    int FUNC_NAME(custatevecHandle_t &handle,                                       \
                  const int nIndexBits,                                             \
                  const int adjoint,                                                \
                  const int target,                                                 \
                  const int controls[],                                             \
                  const int nControls,                                              \
                  PRECISION_TYPE_COMPLEX(selectPrecision) * d_sv,                   \
                  void *extraWorkspace,                                             \
                  size_t &extraWorkspaceSizeInBytes)                                \
    {                                                                               \
        constexpr PRECISION_TYPE_COMPLEX(selectPrecision) matrix[] = MATRIX_VALUES; \
        CHECK_BROAD_ERROR(applyGatesGeneral<selectPrecision>(                       \
            handle,                                                                 \
            nIndexBits,                                                             \
            matrix,                                                                 \
            adjoint,                                                                \
            &target,                                                                \
            1,                                                                      \
            controls,                                                               \
            nControls,                                                              \
            d_sv,                                                                   \
            extraWorkspace,                                                         \
            extraWorkspaceSizeInBytes));                                            \
        return CUSTATEVEC_STATUS_SUCCESS;                                           \
    }                                                                               \
                                                                                    \
    template <precision selectPrecision = precision::bit_64>                        \
    inline int FUNC_NAME(custatevecHandle_t &handle,                                \
                         const int nIndexBits,                                      \
                         const int adjoint,                                         \
                         const int target,                                          \
                         const std::span<const int> &controls,                      \
                         PRECISION_TYPE_COMPLEX(selectPrecision) * d_sv,            \
                         void *extraWorkspace,                                      \
                         size_t &extraWorkspaceSizeInBytes)                         \
    {                                                                               \
        CHECK_BROAD_ERROR(FUNC_NAME<selectPrecision>(                               \
            handle,                                                                 \
            nIndexBits,                                                             \
            adjoint,                                                                \
            target,                                                                 \
            controls.data(),                                                        \
            controls.size(),                                                        \
            d_sv,                                                                   \
            extraWorkspace,                                                         \
            extraWorkspaceSizeInBytes));                                            \
        return CUSTATEVEC_STATUS_SUCCESS;                                           \
    }                                                                               \
    template <precision selectPrecision = precision::bit_64>                        \
    inline int FUNC_NAME(custatevecHandle_t &handle,                                \
                         const int nIndexBits,                                      \
                         const int adjoint,                                         \
                         const std::span<const int> &targets,                       \
                         PRECISION_TYPE_COMPLEX(selectPrecision) * d_sv,            \
                         void *extraWorkspace,                                      \
                         size_t &extraWorkspaceSizeInBytes)                         \
    {                                                                               \
        for (int target : targets)                                                  \
        {                                                                           \
            CHECK_BROAD_ERROR(FUNC_NAME<selectPrecision>(                           \
                handle,                                                             \
                nIndexBits,                                                         \
                adjoint,                                                            \
                target,                                                             \
                d_sv,                                                               \
                extraWorkspace,                                                     \
                extraWorkspaceSizeInBytes));                                        \
        }                                                                           \
        return CUSTATEVEC_STATUS_SUCCESS;                                           \
    }                                                                               \
    template <precision selectPrecision = precision::bit_64>                        \
    inline int FUNC_NAME(custatevecHandle_t &handle,                                \
                         const int nIndexBits,                                      \
                         const int adjoint,                                         \
                         const std::span<const int> &targets,                       \
                         const std::span<const int> &controls,                      \
                         PRECISION_TYPE_COMPLEX(selectPrecision) * d_sv,            \
                         void *extraWorkspace,                                      \
                         size_t &extraWorkspaceSizeInBytes)                         \
    {                                                                               \
        for (int target : targets)                                                  \
        {                                                                           \
            CHECK_BROAD_ERROR(FUNC_NAME<selectPrecision>(                           \
                handle,                                                             \
                nIndexBits,                                                         \
                adjoint,                                                            \
                target,                                                             \
                controls.data(),                                                    \
                controls.size(),                                                    \
                d_sv,                                                               \
                extraWorkspace,                                                     \
                extraWorkspaceSizeInBytes));                                        \
        }                                                                           \
        return CUSTATEVEC_STATUS_SUCCESS;                                           \
    }

DEFINE_GATE_APPLY_FUNCTION(applyH, HMat)
DEFINE_GATE_APPLY_FUNCTION(applyY, YMat)
DEFINE_GATE_APPLY_FUNCTION(applyX, XMat)
DEFINE_GATE_APPLY_FUNCTION(applyZ, ZMat)
