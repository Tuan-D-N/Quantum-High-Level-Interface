#pragma once
#include <type_traits>
#define PRECISION_X_LIST X(bit_32), X(bit_64)

#define X(name) name
enum class precision
{
    PRECISION_X_LIST
};
#undef X

#define PRECISION_TYPE_COMPLEX(selectPrecision) typename std::conditional<selectPrecision == precision::bit_64, cuDoubleComplex, cuComplex>::type
#define PRECISION_TYPE_REAL(selectPrecision) typename std::conditional<selectPrecision == precision::bit_64, double, float>::type

#define PRECISION_VARIABLES_DEFINE                           \
    cudaDataType_t cudaType;                                 \
    custatevecComputeType_t custatevecType;                  \
    if constexpr (SelectPrecision == precision::bit_32)      \
    {                                                        \
        cudaType = CUDA_C_32F;                               \
        custatevecType = CUSTATEVEC_COMPUTE_32F;             \
    }                                                        \
    else if constexpr (SelectPrecision == precision::bit_64) \
    {                                                        \
        cudaType = CUDA_C_64F;                               \
        custatevecType = CUSTATEVEC_COMPUTE_64F;             \
    }
    