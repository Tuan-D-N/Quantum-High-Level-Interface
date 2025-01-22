#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdio>

constexpr double INV_SQRT2(0.707106781186547524400844362105); // Approximation of 1/sqrt(2)
constexpr double SQRT2(1.41421356237309504880168872421);      // Approximation of sqrt(2)

// Macro to check CUDA API errors
#define CHECK_CUDA(func)                                                          \
    {                                                                             \
        cudaError_t status = (func);                                              \
        if (status != cudaSuccess)                                                \
        {                                                                         \
            printf("CUDA API failed at line %d in file %s with error: %s (%d)\n", \
                   __LINE__, __FILE__, cudaGetErrorString(status), status);       \
            return EXIT_FAILURE;                                                  \
        }                                                                         \
    }

// Macro to check cuSPARSE API errors
#define CHECK_CUSPARSE(func)                                                     \
    {                                                                            \
        cusparseStatus_t status = (func);                                        \
        if (status != CUSPARSE_STATUS_SUCCESS)                                   \
        {                                                                        \
            printf("CUSPARSE API failed at line %d in file %s with error: %d\n", \
                   __LINE__, __FILE__, status);                                  \
            return EXIT_FAILURE;                                                 \
        }                                                                        \
    }

// Macro to check cuSPARSE API errors
#define CHECK_CUSTATEVECTOR(func)                                                     \
    {                                                                                 \
        custatevecStatus_t status = (func);                                           \
        if (status != CUSTATEVEC_STATUS_SUCCESS)                                      \
        {                                                                             \
            printf("CUSTATEVECTOR API failed at line %d in file %s with error: %d\n", \
                   __LINE__, __FILE__, status);                                       \
            return EXIT_FAILURE;                                                      \
        }                                                                             \
    }
#define CHECK_BROAD_ERROR(integer)                                    \
    {                                                                 \
        if (integer != 0)                                             \
        {                                                             \
            printf("Broad CUDA ERROR failed at line %d in file %s\n", \
                   __LINE__, __FILE__);                               \
            return EXIT_FAILURE;                                      \
        }                                                             \
    }

// Macro to check CUDA API errors
#define THROW_CUDA(func)                                                          \
    {                                                                             \
        cudaError_t status = (func);                                              \
        if (status != cudaSuccess)                                                \
        {                                                                         \
            printf("CUDA API failed at line %d in file %s with error: %s (%d)\n", \
                   __LINE__, __FILE__, cudaGetErrorString(status), status);       \
            throw std::runtime_error("CUDA API Error");                           \
        }                                                                         \
    }

// Macro to check cuSPARSE API errors
#define THROW_CUSPARSE(func)                                                     \
    {                                                                            \
        cusparseStatus_t status = (func);                                        \
        if (status != CUSPARSE_STATUS_SUCCESS)                                   \
        {                                                                        \
            printf("CUSPARSE API failed at line %d in file %s with error: %d\n", \
                   __LINE__, __FILE__, status);                                  \
            throw std::runtime_error("CUSPARSE API Error");                      \
        }                                                                        \
    }

// Macro to check cuStateVec API errors
#define THROW_CUSTATEVECTOR(func)                                                     \
    {                                                                                 \
        custatevecStatus_t status = (func);                                           \
        if (status != CUSTATEVEC_STATUS_SUCCESS)                                      \
        {                                                                             \
            printf("CUSTATEVECTOR API failed at line %d in file %s with error: %d\n", \
                   __LINE__, __FILE__, status);                                       \
            throw std::runtime_error("CUSTATEVECTOR API Error");                      \
        }                                                                             \
    }
// Macro to check broad errors
#define THROW_BROAD_ERROR(integer)                                    \
    {                                                                 \
        if (integer != 0)                                             \
        {                                                             \
            printf("Broad CUDA ERROR failed at line %d in file %s\n", \
                   __LINE__, __FILE__);                               \
            throw std::runtime_error("Broad CUDA Error");             \
        }                                                             \
    }
