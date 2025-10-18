// test_apply_sparse_gate.cu
#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "../CuSparseControl/SparseGate.hpp"
#include "../CuSparseControl/SparseGateBULK.hpp"

// -----------------------------------------------------------------------------
// Optional ASSERT-style error checking (keeps tests readable/fail-fast)
// -----------------------------------------------------------------------------
#define ASSERT_CUDA_OK(expr)                                      \
    do {                                                          \
        cudaError_t _e = (expr);                                  \
        ASSERT_EQ(_e, cudaSuccess) << "CUDA error: "              \
                                   << cudaGetErrorString(_e);     \
    } while (0)

#define ASSERT_CUSPARSE_OK(expr)                                  \
    do {                                                          \
        cusparseStatus_t _s = (expr);                             \
        ASSERT_EQ(_s, CUSPARSE_STATUS_SUCCESS)                    \
            << "cuSPARSE error (status " << int(_s) << ")";       \
    } while (0)

// -----------------------------------------------------------------------------
// Decls from code under test (headers already included; prototypes here are OK)
// -----------------------------------------------------------------------------
int applySparseGate(
    cusparseHandle_t handle,
    int nQubits,
    const int *d_csrRowPtrU,
    const int *d_csrColIndU,
    const cuDoubleComplex *d_csrValU,
    const cuDoubleComplex *d_state_in,
    cuDoubleComplex *d_state_out,
    const std::vector<int> &targetQubits,
    const std::vector<int> &controlQubits,
    int nnzU);

int applySparseGateBulk(
    cusparseHandle_t handle,
    int nQubits,
    const int *d_csrRowPtrU,
    const int *d_csrColIndU,
    const cuDoubleComplex *d_csrValU,
    const cuDoubleComplex *d_state_in,
    cuDoubleComplex *d_state_out,
    const std::vector<int> &targetQubits,
    const std::vector<int> &controlQubits,
    int nnzU);

// -----------------------------------------------------------------------------
// Helpers (anonymous namespace)
// -----------------------------------------------------------------------------
namespace {

inline cuDoubleComplex C(double r, double i = 0.0) {
    return make_cuDoubleComplex(r, i);
}

void fill_state_host(std::vector<cuDoubleComplex> &h, uint64_t seed = 0xC0FFEEULL) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (auto &z : h) z = C(U(rng), U(rng));
}

void copy_to_device(const std::vector<int> &h, int **d) {
    ASSERT_CUDA_OK(cudaMalloc((void **)d, h.size() * sizeof(int)));
    ASSERT_CUDA_OK(cudaMemcpy(*d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
}
void copy_to_device(const std::vector<cuDoubleComplex> &h, cuDoubleComplex **d) {
    ASSERT_CUDA_OK(cudaMalloc((void **)d, h.size() * sizeof(cuDoubleComplex)));
    ASSERT_CUDA_OK(cudaMemcpy(*d, h.data(), h.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
}

void expect_close(const std::vector<cuDoubleComplex> &a,
                  const std::vector<cuDoubleComplex> &b,
                  double atol = 1e-12, double rtol = 1e-10) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        const double ar = cuCreal(a[i]), ai = cuCimag(a[i]);
        const double br = cuCreal(b[i]), bi = cuCimag(b[i]);
        const double dr = std::abs(ar - br);
        const double di = std::abs(ai - bi);
        const double tol_r = atol + rtol * std::max({std::abs(ar), std::abs(br), 1.0});
        const double tol_i = atol + rtol * std::max({std::abs(ai), std::abs(bi), 1.0});
        if (!(dr <= tol_r && di <= tol_i)) {
            ADD_FAILURE() << "mismatch at i=" << i << ": "
                          << "(" << ar << "," << ai << ") vs (" << br << "," << bi << ") "
                          << "dif=(" << dr << "," << di << ") tol=(" << tol_r << "," << tol_i << ")";
            break;
        }
    }
}

// Build CSR for Pauli-X (2x2): [[0,1],[1,0]]
void build_X_csr(std::vector<int> &rowPtr, std::vector<int> &colInd,
                 std::vector<cuDoubleComplex> &val) {
    rowPtr = {0, 1, 2};
    colInd = {1, 0};
    val    = {C(1), C(1)};
}

// Build CSR for SWAP on 2 qubits (4x4).
// Basis |00>,|01>,|10>,|11| -> rows map to columns [0,2,1,3] with 1s.
void build_SWAP_csr(std::vector<int> &rowPtr, std::vector<int> &colInd,
                    std::vector<cuDoubleComplex> &val) {
    rowPtr = {0, 1, 2, 3, 4};
    colInd = {0, 2, 1, 3};
    val    = {C(1), C(1), C(1), C(1)};
}

// RNG helpers for random tests
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = (x ^ (x >> 31));
    return x;
}
inline double u01(uint64_t &s) {
    s = splitmix64(s);
    return (s >> 11) * (1.0 / 9007199254740992.0); // 53-bit
}
inline cuDoubleComplex rand_cplx(uint64_t &s) {
    const double re = 2.0 * u01(s) - 1.0;
    const double im = 2.0 * u01(s) - 1.0;
    return make_cuDoubleComplex(re, im);
}

// Row-sorted CSR via Bernoulli sampling (no duplicates, naturally sorted)
void make_random_csr_host(int d, double density, uint64_t seed,
                          std::vector<int> &rowPtr,
                          std::vector<int> &colInd,
                          std::vector<cuDoubleComplex> &val) {
    density = std::max(0.0, std::min(1.0, density));
    rowPtr.assign(d + 1, 0);
    colInd.clear();
    val.clear();
    colInd.reserve(size_t(density * d * d * 1.05));
    val.reserve(colInd.capacity());

    uint64_t s = seed;
    for (int r = 0; r < d; ++r) {
        for (int c = 0; c < d; ++c) {
            if (u01(s) < density) {
                colInd.push_back(c);
                val.push_back(rand_cplx(s));
            }
        }
        rowPtr[r + 1] = static_cast<int>(colInd.size());
    }
    // Ensure at least 1 nnz (avoid trivial zero op)
    if (colInd.empty()) {
        colInd.push_back(0);
        val.push_back(make_cuDoubleComplex(1.0, 0.0));
        for (int r = 0; r < d; ++r) rowPtr[r + 1] = 1;
    }
}

// choose k distinct ints from [0..n-1], sorted
std::vector<int> choose_k_sorted(int n, int k, std::mt19937_64 &rng) {
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(k);
    std::sort(all.begin(), all.end());
    return all;
}

std::vector<int> choose_ctrls_excluding(int n, int maxCtrls,
                                        const std::vector<int> &exclude,
                                        std::mt19937_64 &rng) {
    std::vector<char> banned(n, 0);
    for (int x : exclude) banned[x] = 1;

    std::vector<int> pool; pool.reserve(n);
    for (int i = 0; i < n; ++i) if (!banned[i]) pool.push_back(i);
    std::shuffle(pool.begin(), pool.end(), rng);

    const int c = std::min<int>(maxCtrls, static_cast<int>(pool.size()));
    pool.resize(c);
    std::sort(pool.begin(), pool.end());
    return pool;
}

} // namespace

// -----------------------------------------------------------------------------
// Test fixture
// -----------------------------------------------------------------------------
class ApplySparseGateTest : public ::testing::Test {
protected:
    cusparseHandle_t handle = nullptr;

    void SetUp() override {
        ASSERT_CUSPARSE_OK(cusparseCreate(&handle));
    }
    void TearDown() override {
        if (handle) cusparseDestroy(handle);
        handle = nullptr;
    }
};

// -----------------------------------------------------------------------------
// Deterministic tests
// -----------------------------------------------------------------------------

// Case 1: n=5 qubits, target={2}, no controls; U = X (k=1)
TEST_F(ApplySparseGateTest, OneTargetNoControl_PauliX) {
    const int nQubits = 5;
    const size_t dim = size_t(1) << nQubits;

    // CSR for X
    std::vector<int> h_rowPtr, h_colInd;
    std::vector<cuDoubleComplex> h_val;
    build_X_csr(h_rowPtr, h_colInd, h_val);
    const int nnz = static_cast<int>(h_colInd.size());

    // Device CSR
    int *d_rowPtr = nullptr, *d_colInd = nullptr;
    cuDoubleComplex *d_val = nullptr;
    copy_to_device(h_rowPtr, &d_rowPtr);
    copy_to_device(h_colInd, &d_colInd);
    copy_to_device(h_val, &d_val);

    // State
    std::vector<cuDoubleComplex> h_in(dim), h_outA(dim), h_outB(dim);
    fill_state_host(h_in, 12345);

    cuDoubleComplex *d_in = nullptr, *d_outA = nullptr, *d_outB = nullptr;
    ASSERT_CUDA_OK(cudaMalloc(&d_in,   dim * sizeof(cuDoubleComplex)));
    ASSERT_CUDA_OK(cudaMalloc(&d_outA, dim * sizeof(cuDoubleComplex)));
    ASSERT_CUDA_OK(cudaMalloc(&d_outB, dim * sizeof(cuDoubleComplex)));
    ASSERT_CUDA_OK(cudaMemcpy(d_in, h_in.data(), dim * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Targets/controls
    const std::vector<int> targets = {2};
    const std::vector<int> ctrls   = {}; // none

    // Run scalar SpMV loop impl
    ASSERT_EQ(0, applySparseGate(handle, nQubits, d_rowPtr, d_colInd, d_val,
                                 d_in, d_outA, targets, ctrls, nnz));

    // Run bulk SpMM impl
    ASSERT_EQ(0, applySparseGateBulk(handle, nQubits, d_rowPtr, d_colInd, d_val,
                                     d_in, d_outB, targets, ctrls, nnz));

    // Compare
    ASSERT_CUDA_OK(cudaMemcpy(h_outA.data(), d_outA, dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    ASSERT_CUDA_OK(cudaMemcpy(h_outB.data(), d_outB, dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    expect_close(h_outA, h_outB, 1e-12, 1e-10);

    // Cleanup
    cudaFree(d_rowPtr); cudaFree(d_colInd); cudaFree(d_val);
    cudaFree(d_in); cudaFree(d_outA); cudaFree(d_outB);
}

// Case 2: n=6 qubits, targets={1,3}, control={5}; U = SWAP (k=2)
TEST_F(ApplySparseGateTest, TwoTargetsOneControl_SWAP) {
    const int nQubits = 6;
    const size_t dim = size_t(1) << nQubits;

    // CSR for SWAP (4x4)
    std::vector<int> h_rowPtr, h_colInd;
    std::vector<cuDoubleComplex> h_val;
    build_SWAP_csr(h_rowPtr, h_colInd, h_val);
    const int nnz = static_cast<int>(h_colInd.size());

    // Device CSR
    int *d_rowPtr = nullptr, *d_colInd = nullptr;
    cuDoubleComplex *d_val = nullptr;
    copy_to_device(h_rowPtr, &d_rowPtr);
    copy_to_device(h_colInd, &d_colInd);
    copy_to_device(h_val, &d_val);

    // State
    std::vector<cuDoubleComplex> h_in(dim), h_outA(dim), h_outB(dim);
    fill_state_host(h_in, 99999);

    cuDoubleComplex *d_in = nullptr, *d_outA = nullptr, *d_outB = nullptr;
    ASSERT_CUDA_OK(cudaMalloc(&d_in,   dim * sizeof(cuDoubleComplex)));
    ASSERT_CUDA_OK(cudaMalloc(&d_outA, dim * sizeof(cuDoubleComplex)));
    ASSERT_CUDA_OK(cudaMalloc(&d_outB, dim * sizeof(cuDoubleComplex)));
    ASSERT_CUDA_OK(cudaMemcpy(d_in, h_in.data(), dim * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Targets/controls
    const std::vector<int> targets = {1, 3};  // ascending
    const std::vector<int> ctrls   = {5};     // require qubit 5 == 1

    // Run scalar SpMV loop impl
    ASSERT_EQ(0, applySparseGate(handle, nQubits, d_rowPtr, d_colInd, d_val,
                                 d_in, d_outA, targets, ctrls, nnz));

    // Run bulk SpMM impl
    ASSERT_EQ(0, applySparseGateBulk(handle, nQubits, d_rowPtr, d_colInd, d_val,
                                     d_in, d_outB, targets, ctrls, nnz));

    // Compare
    ASSERT_CUDA_OK(cudaMemcpy(h_outA.data(), d_outA, dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    ASSERT_CUDA_OK(cudaMemcpy(h_outB.data(), d_outB, dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    expect_close(h_outA, h_outB, 1e-12, 1e-10);

    // Cleanup
    cudaFree(d_rowPtr); cudaFree(d_colInd); cudaFree(d_val);
    cudaFree(d_in); cudaFree(d_outA); cudaFree(d_outB);
}

// -----------------------------------------------------------------------------
// Randomized stress test
// -----------------------------------------------------------------------------
TEST_F(ApplySparseGateTest, RandomCSR_RandomState_MultipleTrials) {
    // parameter sets
    const std::vector<int>    n_vals   = {4, 5, 6};
    const std::vector<int>    k_vals   = {1, 2};
    const std::vector<double> dens_set = {0.05, 0.20, 0.60, 1.00};

    std::mt19937_64 rng(0xA11CE5EED5ULL);
    const int trials = 6; // random draws per (n,k,density)

    for (int n : n_vals) {
        for (int k : k_vals) {
            if (k >= n) continue; // need at least one non-target qubit
            const int d = 1 << k;
            const size_t dim = size_t(1) << n;

            for (double density : dens_set) {
                for (int t = 0; t < trials; ++t) {
                    // Random targets (sorted), random controls disjoint (0..2)
                    const auto targets = choose_k_sorted(n, k, rng);
                    const auto ctrls   = choose_ctrls_excluding(n, (n > k ? std::min(2, n - k) : 0), targets, rng);

                    // Build random CSR for U (d x d) and copy to device
                    std::vector<int> h_rp, h_ci;
                    std::vector<cuDoubleComplex> h_v;
                    const uint64_t csr_seed =
                        (uint64_t(n) << 48) ^ (uint64_t(k) << 40) ^
                        (uint64_t(t) << 24) ^ (uint64_t)std::llround(density * 1000.0);
                    make_random_csr_host(d, density, csr_seed, h_rp, h_ci, h_v);
                    const int nnzU = static_cast<int>(h_ci.size());

                    int *d_rp = nullptr, *d_ci = nullptr; cuDoubleComplex *d_v = nullptr;
                    copy_to_device(h_rp, &d_rp);
                    copy_to_device(h_ci, &d_ci);
                    copy_to_device(h_v,  &d_v);

                    // Random state
                    std::vector<cuDoubleComplex> h_in(dim);
                    {
                        std::uniform_real_distribution<double> U(-1.0, 1.0);
                        std::mt19937_64 r2(splitmix64(0xBADA55 ^ n ^ (k << 8) ^ t));
                        for (auto &z : h_in) z = make_cuDoubleComplex(U(r2), U(r2));
                    }
                    cuDoubleComplex *d_in = nullptr, *d_outA = nullptr, *d_outB = nullptr;
                    ASSERT_CUDA_OK(cudaMalloc(&d_in,   dim * sizeof(cuDoubleComplex)));
                    ASSERT_CUDA_OK(cudaMalloc(&d_outA, dim * sizeof(cuDoubleComplex)));
                    ASSERT_CUDA_OK(cudaMalloc(&d_outB, dim * sizeof(cuDoubleComplex)));
                    ASSERT_CUDA_OK(cudaMemcpy(d_in, h_in.data(), dim * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

                    // Run both paths
                    ASSERT_EQ(0, applySparseGate(handle, n, d_rp, d_ci, d_v, d_in, d_outA, targets, ctrls, nnzU))
                        << "applySparseGate failed: n=" << n << " k=" << k << " density=" << density << " trial=" << t;

                    ASSERT_EQ(0, applySparseGateBulk(handle, n, d_rp, d_ci, d_v, d_in, d_outB, targets, ctrls, nnzU))
                        << "applySparseGateBulk failed: n=" << n << " k=" << k << " density=" << density << " trial=" << t;

                    // Compare
                    std::vector<cuDoubleComplex> h_outA(dim), h_outB(dim);
                    ASSERT_CUDA_OK(cudaMemcpy(h_outA.data(), d_outA, dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    ASSERT_CUDA_OK(cudaMemcpy(h_outB.data(), d_outB, dim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    expect_close(h_outA, h_outB, 1e-11, 1e-9);

                    // Cleanup
                    cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_v);
                    cudaFree(d_in); cudaFree(d_outA); cudaFree(d_outB);
                }
            }
        }
    }
}
