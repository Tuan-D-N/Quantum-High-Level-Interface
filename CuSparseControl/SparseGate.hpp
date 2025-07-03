#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <c++/11/bits/specfun.h>
#include <cuComplex.h>

#ifdef __CUDACC__
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#else
#define CUDA_KERNEL(...) 
#endif


__global__ void apply_controlled_csr_kernel(
    cuDoubleComplex *state,
    int num_qubits,
    int control_qubit,
    int target_start,
    int num_target_qubits,
    const int *__restrict__ row_ptr,
    const int *__restrict__ col_idx,
    const cuDoubleComplex *__restrict__ values);