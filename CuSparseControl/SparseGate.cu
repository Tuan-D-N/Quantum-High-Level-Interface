#include "SparseGate.hpp"

__global__ void apply_controlled_csr_kernel(
    cuDoubleComplex* state,
    int num_qubits,
    int control_qubit,
    int target_start,
    int num_target_qubits,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const cuDoubleComplex* __restrict__ values
) {
    int total_size = 1 << num_qubits;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int control_mask = 1 << control_qubit;
    if ((idx & control_mask) == 0) return; // Control qubit not active

    int target_mask = ((1 << num_target_qubits) - 1) << target_start;
    int local_idx = (idx & target_mask) >> target_start;
    int base_idx = idx & ~target_mask;

    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (int j = row_ptr[local_idx]; j < row_ptr[local_idx + 1]; ++j) {
        int col = col_idx[j];
        int full_idx = base_idx | (col << target_start);
        cuDoubleComplex val = values[j];
        sum = cuCadd(sum, cuCmul(val, state[full_idx]));
    }

    state[idx] = sum;
}
