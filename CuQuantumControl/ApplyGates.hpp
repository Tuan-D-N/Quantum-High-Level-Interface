#pragma once
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <bitset>
#include "helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR

int applyGatesGeneral(custatevecHandle_t &handle,
                      const int nIndexBits,
                      const cuDoubleComplex matrix[],
                      const int adjoint,
                      const int targets[],
                      const int nTargets,
                      const int controls[],
                      const int nControls,
                      cuDoubleComplex *d_sv,
                      void *extraWorkspace = nullptr,
                      size_t extraWorkspaceSizeInBytes = 0);

#define DEFINE_GATE_APPLY_FUNCTION(FUNC_NAME, MATRIX_VALUES)     \
    custatevecStatus_t FUNC_NAME(custatevecHandle_t &handle,     \
                                 const int nIndexBits,           \
                                 const int adjoint,              \
                                 const int target,               \
                                 cuDoubleComplex *d_sv,          \
                                 void *extraWorkspace = nullptr, \
                                 size_t extraWorkspaceSizeInBytes = 0);


#define HMat                                              \
    {                                                     \
        {INV_SQRT2, 0.0}, {INV_SQRT2, 0.0}, {INV_SQRT2, 0.0}, { -INV_SQRT2, 0.0 } \
    }
DEFINE_GATE_APPLY_FUNCTION(applyH, HMat)

#define XMat                                             \
    {                                                    \
        {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, { 0.0, 0.0 } \
    }
DEFINE_GATE_APPLY_FUNCTION(applyX, XMat)

#define YMat                                              \
    {                                                     \
        {0.0, 0.0}, {0.0, 0.1}, {0.0, -0.1}, { 0.0, 0.0 } \
    }
DEFINE_GATE_APPLY_FUNCTION(applyY, YMat)

#define ZMat                                              \
    {                                                     \
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, { -1.0, 0.0 } \
    }
DEFINE_GATE_APPLY_FUNCTION(applyZ, ZMat)

#define RKMat(k)                                               \
    {                                                          \
        {1.0, 0.0},                                            \
            {0.0, 0.0},                                        \
            {0.0, 0.0},                                        \
        {                                                      \
            cos(2 * M_PI / (1 << k)), sin(2 * M_PI / (1 << k)) \
        }                                                      \
    }
