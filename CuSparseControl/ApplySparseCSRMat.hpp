#include "ApplyMatrixA.hpp"
#include <stdlib.h>           // EXIT_FAILURE
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>
#include <span>
#include <cusparse.h>
#include "../CudaControl/Helper.hpp"
#include "../functionality/WriteAdjMat.hpp"

int applySparseCSRMat(cusparseHandle_t handle,
    std::span<int> csrOffsets,
    std::span<int> csrRows,
    std::span<cuDoubleComplex> values,
    std::span<cuDoubleComplex> svInput,
    std::span<cuDoubleComplex> svOutput);