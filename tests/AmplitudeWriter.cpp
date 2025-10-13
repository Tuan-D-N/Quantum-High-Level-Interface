#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <span>
#include "../CuQuantumControl/AmplitudeWriter.hpp"



// ---- Helpers ----
static inline cuDoubleComplex C(double r, double i = 0.0) {
    return make_cuDoubleComplex(r, i);
}
static inline bool CEq(cuDoubleComplex a, cuDoubleComplex b, double eps=1e-12) {
    return std::abs(cuCreal(a)-cuCreal(b)) <= eps && std::abs(cuCimag(a)-cuCimag(b)) <= eps;
}
static std::vector<cuDoubleComplex> download_state(cuDoubleComplex* d, std::uint64_t nElem) {
    std::vector<cuDoubleComplex> h(nElem);
    cudaMemcpy(h.data(), d, sizeof(cuDoubleComplex) * nElem, cudaMemcpyDeviceToHost);
    return h;
}
static std::uint64_t compose_b_index_from_full(std::uint64_t fullIdx,
                                               const std::vector<std::uint64_t>& targetQubits)
{
    // target bit q contributes to b-index at position "q's order in targetQubits"
    std::uint64_t b = 0ull;
    for (std::uint64_t qpos = 0; qpos < targetQubits.size(); ++qpos) {
        std::uint64_t q = targetQubits[qpos];
        std::uint64_t bit = (fullIdx >> q) & 1ull;
        b |= (bit << qpos);
    }
    return b;
}
static void expect_state_matches_b_across_blocks(const std::vector<cuDoubleComplex>& state,
                                                 std::uint64_t nQubitsTotal,
                                                 const std::vector<cuDoubleComplex>& b,
                                                 const std::vector<std::uint64_t>& targetQubits,
                                                 double eps=1e-12)
{
    const std::uint64_t dim = 1ull << nQubitsTotal;
    for (std::uint64_t idx = 0; idx < dim; ++idx) {
        std::uint64_t bidx = compose_b_index_from_full(idx, targetQubits);
        ASSERT_LT(bidx, b.size());
        EXPECT_TRUE(CEq(state[idx], b[bidx], eps))
            << std::setprecision(16)
            << "Mismatch at index " << idx
            << " (b-index " << bidx << ")\n"
            << "state = (" << cuCreal(state[idx]) << ", " << cuCimag(state[idx]) << ")"
            << ", expected = (" << cuCreal(b[bidx]) << ", " << cuCimag(b[bidx]) << ")";
    }
}

// ===================== Tests =====================

TEST(WriteAmpsU64, N3_Target01_AllBlocks) {
    // n=3, targets = {0,1}. Non-target = {2}. Blocks=2. Size b = 2^2 = 4
    const std::uint64_t n = 3;
    std::vector<std::uint64_t> targets{0,1};
    std::vector<cuDoubleComplex> b = { C(1.0,0.1), C(2.0,0.2), C(3.0,0.3), C(4.0,0.4) };

    const std::uint64_t dim = 1ull << n;
    cuDoubleComplex* d_state = nullptr;
    ASSERT_EQ(cudaMalloc((void**)&d_state, sizeof(cuDoubleComplex) * dim), cudaSuccess);

    // init with sentinel (will be overwritten everywhere for this mapping)
    std::vector<cuDoubleComplex> init(dim, C(999.0, -999.0));
    ASSERT_EQ(cudaMemcpy(d_state, init.data(), sizeof(cuDoubleComplex) * dim, cudaMemcpyHostToDevice), cudaSuccess);

    write_amplitudes_to_target_qubits_u64(d_state, n, std::span<const cuDoubleComplex>(b), std::span<const std::uint64_t>(targets));

    auto h = download_state(d_state, dim);
    expect_state_matches_b_across_blocks(h, n, b, targets, 1e-12);

    cudaFree(d_state);
}

TEST(WriteAmpsU64, N4_Target13_NonContiguous) {
    // n=4, targets = {1,3}. Non-target = {0,2}. Blocks=4. b size = 4.
    const std::uint64_t n = 4;
    std::vector<std::uint64_t> targets{1,3};
    std::vector<cuDoubleComplex> b = { C(0.5,0.0), C(0.0,0.5), C(-0.5,0.1), C(0.1,-0.5) };

    const std::uint64_t dim = 1ull << n;
    cuDoubleComplex* d_state = nullptr;
    ASSERT_EQ(cudaMalloc((void**)&d_state, sizeof(cuDoubleComplex) * dim), cudaSuccess);

    std::vector<cuDoubleComplex> init(dim, C(-7.0, 3.0));
    ASSERT_EQ(cudaMemcpy(d_state, init.data(), sizeof(cuDoubleComplex) * dim, cudaMemcpyHostToDevice), cudaSuccess);

    write_amplitudes_to_target_qubits_u64(d_state, n, std::span<const cuDoubleComplex>(b), std::span<const std::uint64_t>(targets));

    auto h = download_state(d_state, dim);
    expect_state_matches_b_across_blocks(h, n, b, targets, 1e-12);

    cudaFree(d_state);
}

TEST(WriteAmpsU64, N5_SingleTargetBit_ReplicatesAcrossAllBlocks) {
    // n=5, single target qubit {4}. b size=2, replicated across all 2^(n-1)=16 blocks.
    const std::uint64_t n = 5;
    std::vector<std::uint64_t> targets{4};
    std::vector<cuDoubleComplex> b = { C(0.123, 0.0), C(-0.987, 0.25) };

    const std::uint64_t dim = 1ull << n;
    cuDoubleComplex* d_state = nullptr;
    ASSERT_EQ(cudaMalloc((void**)&d_state, sizeof(cuDoubleComplex) * dim), cudaSuccess);

    std::vector<cuDoubleComplex> init(dim, C(42.0, 42.0));
    ASSERT_EQ(cudaMemcpy(d_state, init.data(), sizeof(cuDoubleComplex) * dim, cudaMemcpyHostToDevice), cudaSuccess);

    write_amplitudes_to_target_qubits_u64(d_state, n, std::span<const cuDoubleComplex>(b), std::span<const std::uint64_t>(targets));

    auto h = download_state(d_state, dim);
    expect_state_matches_b_across_blocks(h, n, b, targets, 1e-12);

    cudaFree(d_state);
}

TEST(WriteAmpsU64, TargetOrderDefinesBitOrdering) {
    // Verify that the ORDER of targetQubits defines which |b> bit controls which qubit.
    // n=3, targets={2,0} (notice order 2 then 0). Then b-index bit0 maps to qubit 2, bit1 maps to qubit 0.
    const std::uint64_t n = 3;
    std::vector<std::uint64_t> targets{2,0};
    // b[0]=a00, b[1]=a10 (bit0 set -> qubit 2), b[2]=a01 (bit1 set -> qubit 0), b[3]=a11
    std::vector<cuDoubleComplex> b = { C(10,0), C(20,0), C(30,0), C(40,0) };

    const std::uint64_t dim = 1ull << n;
    cuDoubleComplex* d_state = nullptr;
    ASSERT_EQ(cudaMalloc((void**)&d_state, sizeof(cuDoubleComplex) * dim), cudaSuccess);
    std::vector<cuDoubleComplex> init(dim, C(0,0));
    ASSERT_EQ(cudaMemcpy(d_state, init.data(), sizeof(cuDoubleComplex) * dim, cudaMemcpyHostToDevice), cudaSuccess);

    write_amplitudes_to_target_qubits_u64(d_state, n, std::span<const cuDoubleComplex>(b), std::span<const std::uint64_t>(targets));

    auto h = download_state(d_state, dim);

    // Manually spot-check several indices
    // Full index bits are [q2 q1 q0]. Targets are q2(bit0 of b), q0(bit1 of b).
    auto expect_b = [&](std::uint64_t fullIdx)->std::uint64_t{
        // bidx bit0 = fullIdx@q2; bidx bit1 = fullIdx@q0
        std::uint64_t b0 = (fullIdx >> 2) & 1ull;
        std::uint64_t b1 = (fullIdx >> 0) & 1ull;
        return (b0 << 0) | (b1 << 1);
    };
    for (std::uint64_t idx : {0ull,1ull,2ull,3ull,4ull,5ull,6ull,7ull}) {
        std::uint64_t bidx = expect_b(idx);
        EXPECT_TRUE(CEq(h[idx], b[bidx], 1e-12))
            << "idx="<<idx<<" bidx="<<bidx
            << " state=("<<cuCreal(h[idx])<<","<<cuCimag(h[idx])<<")"
            << " expected=("<<cuCreal(b[bidx])<<","<<cuCimag(b[bidx])<<")";
    }

    cudaFree(d_state);
}
