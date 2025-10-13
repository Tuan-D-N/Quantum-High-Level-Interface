#pragma once
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <custatevec.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <bitset>
#include "../CudaControl/Helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR
#include <cusparse.h>
#include <vector>
#include <cmath>
#include <span>
#include <optional>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cassert>

// --------- host API ---------

/**
 * @brief Normalize a device statevector so that sum |z_i|^2 = 1.
 *
 * @param d_sv       [device] pointer to cuDoubleComplex array
 * @param length     number of elements (uint64_t)
 * @param out_norm2  optional host pointer; if non-null, receives the pre-normalization ||v||^2
 *
 * @return cudaError_t from the last CUDA runtime call (cudaSuccess on success)
 */
cudaError_t square_normalize_statevector_u64(
    cuDoubleComplex *d_sv,
    std::uint64_t length,
    double *out_norm2 = nullptr);