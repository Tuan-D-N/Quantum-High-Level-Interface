#pragma once
#include <cuComplex.h>

#define INV_SQRT2 (0.7071067811865475) // Approximation of 1/sqrt(2)

#define HANDLE_ERROR(x)                                                        \
{   const auto err = x;                                                        \
    if (err != CUSTATEVEC_STATUS_SUCCESS ) {                                   \
        printf("Error custatevec: %s in line %d\n",                                       \
               custatevecGetErrorString(err), __LINE__); return err; }         \
};

#define HANDLE_CUDA_ERROR(x)                                                   \
{   const auto err = x;                                                        \
    if (err != cudaSuccess ) {                                                 \
        printf("Error cuda: %s in line %d\n",                                       \
               cudaGetErrorString(err), __LINE__); return err; }               \
};

