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

// Writes amplitudes_b onto the subspace spanned by targetQubits,
// but ONLY for those basis indices where all *non-target* qubits
// match ANY of the provided bitmap patterns.
// Each pattern p is given by (maskOrderingQubitsVec[p], maskBitStringsVec[p]).
// - maskOrderingQubitsVec[p][i] is a qubit index
// - maskBitStringsVec[p][i] is the required 0/1 value at that qubit
//
// Requirements (asserted):
//   * nQubitsTotal <= 63
//   * amplitudes_b.size() == 2^(|targetQubits|)
//   * maskOrderingQubitsVec.size() == maskBitStringsVec.size()
//   * For every pattern p: sizes match; values are 0/1; qubits in-range
//   * Patterns are disjoint from targetQubits (to avoid ambiguity)
//
// Notes:
//   * scatter_kernel<<<grid, TPB>>>(d_state, d_b, d_offsets, size)
//     is assumed to exist and to assign d_state[offsets[i]] = d_b[i].
void write_amplitudes_to_target_qubits_u64_masked(
    cuDoubleComplex* d_state,
    uint64_t nQubitsTotal,
    std::span<const cuDoubleComplex> amplitudes_b,
    std::span<const uint64_t> targetQubits,
    const std::vector<std::vector<int>>& maskOrderingQubitsVec,
    const std::vector<std::vector<int>>& maskBitStringsVec)
{
    // ---- sanity checks ----
    assert(d_state != nullptr);
    assert(nQubitsTotal > 0);
    assert(nQubitsTotal <= 63);

    const uint64_t numberOfTargets = targetQubits.size();
    assert(numberOfTargets <= nQubitsTotal);

    const uint64_t sizeOfTargetVector = (numberOfTargets == 0) ? 1ull : (1ull << numberOfTargets);
    assert(amplitudes_b.size() == sizeOfTargetVector);

    // pattern vectors must align
    assert(maskOrderingQubitsVec.size() == maskBitStringsVec.size());

    // Build a quick lookup set for targetQubits to assert disjointness
    std::vector<bool> isTarget(nQubitsTotal, false);
    for (uint64_t q : targetQubits) {
        assert(q < nQubitsTotal);
        assert(!isTarget[q]);
        isTarget[q] = true;
    }

    // Validate patterns; also precompute each pattern’s fixed base mask
    const size_t nPatterns = maskOrderingQubitsVec.size();
    std::vector<uint64_t> baseMasks(nPatterns, 0ull);

    for (size_t p = 0; p < nPatterns; ++p) {
        const auto& ord = maskOrderingQubitsVec[p];
        const auto& bits = maskBitStringsVec[p];
        assert(ord.size() == bits.size());

        uint64_t base = 0ull;
        for (size_t i = 0; i < ord.size(); ++i) {
            int qi = ord[i];
            int vi = bits[i];
            assert(qi >= 0 && static_cast<uint64_t>(qi) < nQubitsTotal);
            assert(vi == 0 || vi == 1);
            // Must not constrain a target qubit (to keep semantics unambiguous)
            assert(!isTarget[static_cast<uint64_t>(qi)]);

            if (vi == 1) base |= (1ull << static_cast<uint64_t>(qi));
        }
        baseMasks[p] = base;
    }

    // ---- upload |b| once ----
    cuDoubleComplex* d_b = nullptr;
    cudaMalloc((void**)&d_b, sizeof(cuDoubleComplex) * sizeOfTargetVector);
    cudaMemcpy(d_b, amplitudes_b.data(),
               sizeof(cuDoubleComplex) * sizeOfTargetVector,
               cudaMemcpyHostToDevice);

    // ---- offsets buffer (host + device) as 64-bit ----
    std::vector<uint64_t> h_offsets(sizeOfTargetVector);

    // Kernel expects unsigned long long* for 64-bit indices
    unsigned long long* d_offsets = nullptr;
    cudaMalloc((void**)&d_offsets, sizeof(unsigned long long) * sizeOfTargetVector);

    // ---- kernel launch config ----
    const unsigned int TPB = 256u;
    assert(((sizeOfTargetVector + TPB - 1u) / TPB) <= std::numeric_limits<unsigned int>::max());
    const unsigned int grid = static_cast<unsigned int>((sizeOfTargetVector + TPB - 1u) / TPB);

    // ---- helper: build target mask for a given b ----
    auto make_target_mask = [&](uint64_t b)->uint64_t {
        uint64_t m = 0ull;
        for (uint64_t q = 0; q < numberOfTargets; ++q) {
            if ((b >> q) & 1ull) {
                const uint64_t tq = targetQubits[q];
                // already validated in isTarget
                m |= (1ull << tq);
            }
        }
        return m;
    };

    // ---- for each pattern, scatter |b| at offsets = baseMask | targetMask ----
    for (size_t p = 0; p < nPatterns; ++p) {
        const uint64_t baseMask = baseMasks[p];

        // fill offsets for all target basis indices (b = 0..2^k-1)
        for (uint64_t b = 0; b < sizeOfTargetVector; ++b) {
            h_offsets[b] = baseMask | make_target_mask(b);
        }

        // upload offsets and scatter
        cudaMemcpy(d_offsets, h_offsets.data(),
                   sizeof(unsigned long long) * sizeOfTargetVector,
                   cudaMemcpyHostToDevice);

        scatter_kernel CUDA_KERNEL(grid, TPB)(d_state, d_b, d_offsets, sizeOfTargetVector);
    }

    cudaDeviceSynchronize();
    cudaFree(d_offsets);
    cudaFree(d_b);
}
