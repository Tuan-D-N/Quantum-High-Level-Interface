#pragma once
#include <stdio.h>


#define INV_SQRT2 (0.7071067811865475) // Approximation of 1/sqrt(2)



// Macro to check CUDA API errors
#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

// Macro to check cuSPARSE API errors
#define CHECK_CUSPARSE(func)                                          \
    {                                                                 \
        cusparseStatus_t status = (func);                             \
        if (status != CUSPARSE_STATUS_SUCCESS)                        \
        {                                                             \
            printf("CUSPARSE API failed at line %d with error: %d\n", \
                   __LINE__, status);                                 \
            return EXIT_FAILURE;                                      \
        }                                                             \
    }

// Macro to check cuSPARSE API errors
#define CHECK_CUSTATEVECTOR(func)                                     \
    {                                                                 \
        custatevecStatus_t status = (func);                           \
        if (status != CUSTATEVEC_STATUS_SUCCESS)                      \
        {                                                             \
            printf("CUSPARSE API failed at line %d with error: %d\n", \
                   __LINE__, status);                                 \
            return EXIT_FAILURE;                                      \
        }                                                             \
    }

#define CHECK_BROAD_ERROR(integer)                         \
    {                                                      \
        if (integer != 0)                                  \
        {                                                  \
            printf("Broad CUDA ERROR failed at line %d\n", \
                   __LINE__);                              \
            return EXIT_FAILURE;                           \
        }                                                  \
    }
