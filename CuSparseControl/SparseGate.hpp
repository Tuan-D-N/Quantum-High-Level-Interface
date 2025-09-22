#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <c++/11/bits/specfun.h>
#include <cuComplex.h>
#include <vector>
#include <cusparse.h>
#include <stdlib.h>           // EXIT_FAILURE
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.

#ifdef __CUDACC__
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#else
#define CUDA_KERNEL(...) 
#endif


int applySparseGate(
    cusparseHandle_t handle,
    int nQubits,
    const int* d_csrRowPtrU,
    const int* d_csrColIndU,
    const cuDoubleComplex* d_csrValU,
    const cuDoubleComplex* d_state_in,
    cuDoubleComplex* d_state_out,
    const std::vector<int>& targetQubits,
    int nnzU);
