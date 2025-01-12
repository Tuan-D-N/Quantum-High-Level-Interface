#pragma once
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <array>
#include <bitset>
#include "../CudaControl/Helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR

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
                      size_t &extraWorkspaceSizeInBytes);

#define DEFINE_GATE_APPLY_FUNCTION(FUNC_NAME, MATRIX_VALUES)         \
    int FUNC_NAME(custatevecHandle_t &handle,                        \
                  const int nIndexBits,                              \
                  const int adjoint,                                 \
                  const int target,                                  \
                  cuDoubleComplex *d_sv,                             \
                  void *extraWorkspace,                              \
                  size_t &extraWorkspaceSizeInBytes);                \
    int FUNC_NAME(custatevecHandle_t &handle,                        \
                  const int nIndexBits,                              \
                  const int adjoint,                                 \
                  const int target,                                  \
                  const int controls[],                              \
                  const int nControls,                               \
                  cuDoubleComplex *d_sv,                             \
                  void *extraWorkspace,                              \
                  size_t &extraWorkspaceSizeInBytes);                \
    template <int n>                                                 \
    inline int FUNC_NAME(custatevecHandle_t &handle,                 \
                         const int nIndexBits,                       \
                         const int adjoint,                          \
                         const int target,                           \
                         const std::array<int, n> &controls,         \
                         cuDoubleComplex *d_sv,                      \
                         void *extraWorkspace,                       \
                         size_t &extraWorkspaceSizeInBytes)          \
    {                                                                \
        CHECK_BROAD_ERROR(FUNC_NAME(                                 \
            handle,                                                  \
            nIndexBits,                                              \
            adjoint,                                                 \
            target,                                                  \
            controls.data(),                                         \
            controls.size(),                                         \
            d_sv,                                                    \
            extraWorkspace,                                          \
            extraWorkspaceSizeInBytes));                             \
        return CUSTATEVEC_STATUS_SUCCESS;                            \
    }                                                                \
    template <int n>                                                 \
    inline int FUNC_NAME(custatevecHandle_t &handle,                 \
                         const int nIndexBits,                       \
                         const int adjoint,                          \
                         const std::array<int, n> &targets,          \
                         cuDoubleComplex *d_sv,                      \
                         void *extraWorkspace,                       \
                         size_t &extraWorkspaceSizeInBytes)          \
    {                                                                \
        for (int target : targets)                                   \
        {                                                            \
            CHECK_BROAD_ERROR(FUNC_NAME(                             \
                handle,                                              \
                nIndexBits,                                          \
                adjoint,                                             \
                target,                                              \
                d_sv,                                                \
                extraWorkspace,                                      \
                extraWorkspaceSizeInBytes));                         \
        }                                                            \
        return CUSTATEVEC_STATUS_SUCCESS;                            \
    }                                                                \
    template <int n_target, int n_control>                           \
    inline int FUNC_NAME(custatevecHandle_t &handle,                 \
                         const int nIndexBits,                       \
                         const int adjoint,                          \
                         const std::array<int, n_target> &targets,   \
                         const std::array<int, n_control> &controls, \
                         cuDoubleComplex *d_sv,                      \
                         void *extraWorkspace,                       \
                         size_t &extraWorkspaceSizeInBytes)          \
    {                                                                \
        for (int target : targets)                                   \
        {                                                            \
            CHECK_BROAD_ERROR(FUNC_NAME(                             \
                handle,                                              \
                nIndexBits,                                          \
                adjoint,                                             \
                target,                                              \
                controls.data(),                                     \
                controls.size(),                                     \
                d_sv,                                                \
                extraWorkspace,                                      \
                extraWorkspaceSizeInBytes));                         \
        }                                                            \
        return CUSTATEVEC_STATUS_SUCCESS;                            \
    }

#define HMat                                                                      \
    {                                                                             \
        {INV_SQRT2, 0.0}, {INV_SQRT2, 0.0}, {INV_SQRT2, 0.0}, { -INV_SQRT2, 0.0 } \
    }

#define XMat                                             \
    {                                                    \
        {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, { 0.0, 0.0 } \
    }

#define YMat                                              \
    {                                                     \
        {0.0, 0.0}, {0.0, 0.1}, {0.0, -0.1}, { 0.0, 0.0 } \
    }

#define ZMat                                              \
    {                                                     \
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, { -1.0, 0.0 } \
    }

#define RKMat(k)                                               \
    {                                                          \
        {1.0, 0.0},                                            \
            {0.0, 0.0},                                        \
            {0.0, 0.0},                                        \
        {                                                      \
            cos(2 * M_PI / (1 << k)), sin(2 * M_PI / (1 << k)) \
        }                                                      \
    }

#define RXMat(theta)                \
    {                               \
        {cos(theta / 2), 0.0},      \
            {0.0, -sin(theta / 2)}, \
            {0.0, -sin(theta / 2)}, \
        {                           \
            cos(theta / 2), 0.0     \
        }                           \
    }

#define RYMat(theta)                \
    {                               \
        {cos(theta / 2), 0.0},      \
            {-sin(theta / 2), 0.0}, \
            {sin(theta / 2), 0.0},  \
        {                           \
            cos(theta / 2), 0.0     \
        }                           \
    }

#define RZMat(theta)                        \
    {                                       \
        {cos(-theta / 2), sin(-theta / 2)}, \
            {0.0, 0.0},                     \
            {0.0, 0.0},                     \
        {                                   \
            cos(theta / 2), sin(theta / 2)  \
        }                                   \
    }

DEFINE_GATE_APPLY_FUNCTION(applyH, HMat)
DEFINE_GATE_APPLY_FUNCTION(applyY, YMat)
DEFINE_GATE_APPLY_FUNCTION(applyZ, ZMat)
DEFINE_GATE_APPLY_FUNCTION(applyX, XMat)