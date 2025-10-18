#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <device_launch_parameters.h>
#include "../CudaControl/Helper.hpp"
#include "../CuSparseControl/SparseHelper.hpp"
#include <span>
#include <cassert>
#include <cstdint> // std::uint64_t

/**
 * Write a small state |b> (amplitudes) into the subspace spanned by targetQubits
 * across *all* assignments of the non-target qubits.
 *
 * Indexing convention: bit q of a basis index corresponds to qubit q.
 *
 * @param d_state           [device] full statevector of size 2^n (caller owns allocation/size)
 * @param nQubitsTotal      n (total number of qubits in d_state)
 * @param amplitudes_b      host span of size 2^k (k = targetQubits.size())
 * @param targetQubits      host span of k distinct qubit indices in [0, nQubitsTotal)
 */
void write_amplitudes_to_target_qubits_u64(
    cuDoubleComplex* d_state,
    uint64_t nQubitsTotal,
    std::span<const cuDoubleComplex> amplitudes_b,
    std::span<const uint64_t> targetQubits);


void write_amplitudes_to_target_qubits_u64_masked(
    cuDoubleComplex* d_state,
    uint64_t nQubitsTotal,
    std::span<const cuDoubleComplex> amplitudes_b,
    std::span<const uint64_t> targetQubits,
    const std::vector<std::vector<int>>& maskOrderingQubitsVec,
    const std::vector<std::vector<int>>& maskBitStringsVec);