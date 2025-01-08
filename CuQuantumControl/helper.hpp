#pragma once
#include <cuComplex.h>

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

bool almost_equal(cuDoubleComplex x, cuDoubleComplex y);

bool almost_equal(double x, double y);