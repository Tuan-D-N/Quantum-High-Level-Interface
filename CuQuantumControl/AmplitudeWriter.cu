#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <device_launch_parameters.h>
#include "../CudaControl/Helper.hpp"
#include "../CuSparseControl/SparseHelper.hpp"
#include "AmplitudeWriter.hpp"
#include <span>
#include <cassert>
#include <cstdint> // std::uint64_t


void write_amplitudes_to_target_qubits_u64(
    cuDoubleComplex* d_state,
    uint64_t nQubitsTotal,
    std::span<const cuDoubleComplex> amplitudes_b,
    std::span<const uint64_t> targetQubits)
{
    // ---- sanity checks ----
    assert(d_state != nullptr);
    assert(nQubitsTotal > 0);
    // we store bitmasks in 64 bits -> require n <= 63 (sign bit unused)
    assert(nQubitsTotal <= 63);

    uint64_t numberOfTargets = targetQubits.size();
    assert(numberOfTargets <= nQubitsTotal);

    uint64_t sizeOfTargetVector = 1ull << numberOfTargets;     // 2^k
    assert(amplitudes_b.size() == sizeOfTargetVector);

    // ---- build nonTargetQubits = [0..n-1] \ targetQubits ----
    std::vector<bool> isTarget;
    isTarget.resize(nQubitsTotal, false);
    for (uint64_t q : targetQubits) {
        assert(q < nQubitsTotal);
        isTarget[q] = true;
    }

    std::vector<uint64_t> nonTargetQubits;
    nonTargetQubits.reserve(nQubitsTotal - numberOfTargets);
    for (uint64_t q = 0; q < nQubitsTotal; ++q) {
        if (!isTarget[q]) nonTargetQubits.push_back(q);
    }

    uint64_t nNon    = nonTargetQubits.size();
    uint64_t nBlocks = (nNon == 0) ? 1ull : (1ull << nNon);

    // ---- upload |b| once ----
    cuDoubleComplex* d_b = nullptr;
    cudaMalloc((void**)&d_b, sizeof(cuDoubleComplex) * sizeOfTargetVector);
    cudaMemcpy(d_b, amplitudes_b.data(),
               sizeof(cuDoubleComplex) * sizeOfTargetVector,
               cudaMemcpyHostToDevice);

    // ---- offsets (host + device) as uint64_t ----
    std::vector<uint64_t> h_offsets;
    h_offsets.resize(sizeOfTargetVector);

    unsigned long long* d_offsets = nullptr;
    cudaMalloc((void**)&d_offsets, sizeof(unsigned long long) * sizeOfTargetVector);

    // ---- kernel launch config ----
    const unsigned int TPB = 256u;
    // ensure grid size fits into 32-bit launch parameter
    assert(((sizeOfTargetVector + TPB - 1u) / TPB) <= std::numeric_limits<unsigned int>::max());
    unsigned int grid = (sizeOfTargetVector + TPB - 1u) / TPB;

    // === main loop over all non-target assignments ===
    for (uint64_t blk = 0; blk < nBlocks; ++blk)
    {
        // nonTargetMask for this block
        uint64_t nonTargetMask = 0ull;
        for (uint64_t i = 0; i < nonTargetQubits.size(); ++i) {
            if ((blk >> i) & 1ull)
                nonTargetMask |= (1ull << nonTargetQubits[i]);
        }

        // fill offsets for all target basis indices
        for (uint64_t b = 0; b < sizeOfTargetVector; ++b)
        {
            uint64_t targetMask = 0ull;
            for (uint64_t q = 0; q < numberOfTargets; ++q) {
                if ((b >> q) & 1ull)
                    targetMask |= (1ull << targetQubits[q]);
            }
            h_offsets[b] = nonTargetMask | targetMask;
        }

        // upload offsets and scatter |b| into those indices
        cudaMemcpy(d_offsets, h_offsets.data(),
                   sizeof(unsigned long long) * sizeOfTargetVector,
                   cudaMemcpyHostToDevice);

        scatter_kernel CUDA_KERNEL(grid, TPB)(d_state, d_b, d_offsets, sizeOfTargetVector);
    }

    cudaDeviceSynchronize();
    cudaFree(d_offsets);
    cudaFree(d_b);
}