#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <random>
#include <span>
#include <optional>
#include <iomanip>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>

#include "../CuQuantumControl/Normalise_d.hpp"
#include <cstdint>
#include <cmath>

// ---- Function under test (include your header if available) ----
extern cudaError_t square_normalize_statevector_u64(
    cuDoubleComplex* d_sv,
    std::uint64_t length,
    double* out_norm2 /*optional*/);

// ---- Helpers ----
static inline cuDoubleComplex C(double r, double i=0.0) {
    return make_cuDoubleComplex(r, i);
}
static inline bool CEq(cuDoubleComplex a, cuDoubleComplex b, double eps=1e-12) {
    return std::abs(cuCreal(a)-cuCreal(b)) <= eps && std::abs(cuCimag(a)-cuCimag(b)) <= eps;
}
static double host_norm2(const std::vector<cuDoubleComplex>& v) {
    long double acc = 0.0L;
    for (auto z : v) {
        long double re = cuCreal(z);
        long double im = cuCimag(z);
        acc += re*re + im*im;
    }
    return static_cast<double>(acc);
}
static std::vector<cuDoubleComplex> download_vec(cuDoubleComplex* d, std::uint64_t n) {
    std::vector<cuDoubleComplex> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, sizeof(cuDoubleComplex)*n, cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}
static void upload_vec(const std::vector<cuDoubleComplex>& h, cuDoubleComplex** d_out) {
    ASSERT_EQ(cudaMalloc((void**)d_out, sizeof(cuDoubleComplex)*h.size()), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(*d_out, h.data(), sizeof(cuDoubleComplex)*h.size(), cudaMemcpyHostToDevice), cudaSuccess);
}

// ========================== Tests ==========================

TEST(SquareNormalizeU64, SmallDeterministicVector) {
    // length = 8 with mixed complex values
    const std::uint64_t n = 8;
    std::vector<cuDoubleComplex> h(n);
    h[0]=C(1,0); h[1]=C(2,1); h[2]=C(-3,0.5); h[3]=C(0,-4);
    h[4]=C(0.25,-0.75); h[5]=C(1.5,2.5); h[6]=C(-0.125,0.0); h[7]=C(0.0,3.0);

    const double norm2_before = host_norm2(h);
    ASSERT_GT(norm2_before, 0.0);

    cuDoubleComplex* d=nullptr;
    upload_vec(h, &d);

    double norm2_reported = -1.0;
    ASSERT_EQ(square_normalize_statevector_u64(d, n, &norm2_reported), cudaSuccess);
    EXPECT_NEAR(norm2_reported, norm2_before, 1e-12);

    auto hn = download_vec(d, n);
    // Check sum |.|^2 == 1
    EXPECT_NEAR(host_norm2(hn), 1.0, 1e-12);

    // Check proportionality: hn[i] ≈ h[i] / sqrt(norm2_before)
    const double alpha = 1.0/std::sqrt(norm2_before);
    for (std::uint64_t i=0;i<n;++i) {
        cuDoubleComplex expi = C(alpha*cuCreal(h[i]), alpha*cuCimag(h[i]));
        EXPECT_TRUE(CEq(hn[i], expi, 5e-13))
            << std::setprecision(16)
            << "i="<<i<<" got=("<<cuCreal(hn[i])<<","<<cuCimag(hn[i])<<") "
            << "exp=("<<cuCreal(expi)<<","<<cuCimag(expi)<<")";
    }

    cudaFree(d);
}

TEST(SquareNormalizeU64, RandomLargeVector_StridingAndBlocks) {
    // Large enough for many blocks & striding
    const std::uint64_t n = 1ull << 20; // ~1M elements
    std::mt19937_64 rng(123);
    std::normal_distribution<double> N(0.0, 1.0);

    std::vector<cuDoubleComplex> h(n);
    for (auto& z : h) z = C(N(rng), N(rng));

    const double norm2_before = host_norm2(h);
    ASSERT_GT(norm2_before, 0.0);

    cuDoubleComplex* d=nullptr;
    upload_vec(h, &d);

    double norm2_reported = -1.0;
    ASSERT_EQ(square_normalize_statevector_u64(d, n, &norm2_reported), cudaSuccess);
    EXPECT_NEAR(norm2_reported, norm2_before, 1e-6); // reduction order -> looser tol OK for large n

    auto hn = download_vec(d, n);
    EXPECT_NEAR(host_norm2(hn), 1.0, 1e-9);

    cudaFree(d);
}

TEST(SquareNormalizeU64, ZeroVector_NoChange_NoNan) {
    const std::uint64_t n = 4096;
    std::vector<cuDoubleComplex> h(n, C(0.0,0.0));

    cuDoubleComplex* d=nullptr;
    upload_vec(h, &d);

    double norm2_reported = 123.0;
    ASSERT_EQ(square_normalize_statevector_u64(d, n, &norm2_reported), cudaSuccess);
    EXPECT_NEAR(norm2_reported, 0.0, 0.0); // exactly zero

    auto hn = download_vec(d, n);
    // Still all zeros; sum is zero (function leaves as-is)
    EXPECT_NEAR(host_norm2(hn), 0.0, 0.0);

    cudaFree(d);
}

TEST(SquareNormalizeU64, AlreadyNormalized_VectorStaysSame) {
    const std::uint64_t n = 10000; // non power-of-two length (also covers bounds handling)
    std::mt19937_64 rng(999);
    std::normal_distribution<double> N(0.0,1.0);

    std::vector<cuDoubleComplex> h(n);
    long double acc=0.0L;
    for (auto& z : h) {
        double re=N(rng), im=N(rng);
        acc += re*re + im*im;
        z = C(re,im);
    }
    const double alpha = 1.0/std::sqrt(static_cast<double>(acc));
    for (auto& z : h) z = C(alpha*cuCreal(z), alpha*cuCimag(z));

    // sanity: already normalized
    EXPECT_NEAR(host_norm2(h), 1.0, 1e-12);

    cuDoubleComplex* d=nullptr;
    upload_vec(h, &d);

    double norm2_reported = -1.0;
    ASSERT_EQ(square_normalize_statevector_u64(d, n, &norm2_reported), cudaSuccess);
    EXPECT_NEAR(norm2_reported, 1.0, 1e-12);

    auto hn = download_vec(d, n);
    // Expect no change (within rounding from second pass)
    for (std::uint64_t i=0;i<n;++i) {
        EXPECT_TRUE(CEq(hn[i], h[i], 1e-12))
            << std::setprecision(16)
            << "i="<<i<<" got=("<<cuCreal(hn[i])<<","<<cuCimag(hn[i])<<") "
            << "orig=("<<cuCreal(h[i])<<","<<cuCimag(h[i])<<")";
    }
    EXPECT_NEAR(host_norm2(hn), 1.0, 1e-12);

    cudaFree(d);
}

TEST(SquareNormalizeU64, LengthNotMultipleOfBlockSize) {
    // Ensure kernel bounds check works when len % TPB != 0
    const std::uint64_t n = 12345;
    std::vector<cuDoubleComplex> h(n);
    for (std::uint64_t i=0;i<n;++i) {
        double re = 0.001 * double(i+1);
        double im = 0.002 * double((i+3) % 11);
        h[i] = C(re, im);
    }
    const double norm2_before = host_norm2(h);
    ASSERT_GT(norm2_before, 0.0);

    cuDoubleComplex* d=nullptr;
    upload_vec(h, &d);

    double norm2_reported = -1.0;
    ASSERT_EQ(square_normalize_statevector_u64(d, n, &norm2_reported), cudaSuccess);
    EXPECT_NEAR(norm2_reported, norm2_before, 1e-12);

    auto hn = download_vec(d, n);
    EXPECT_NEAR(host_norm2(hn), 1.0, 1e-12);

    // Proportionality spot-check for a few indices
    const double alpha = 1.0/std::sqrt(norm2_before);
    std::array<unsigned long, 5> idxes {0ull, 1ull, 17ull, 1024ull, n-1};
    for (auto idx : idxes) {
        cuDoubleComplex expi = C(alpha*cuCreal(h[idx]), alpha*cuCimag(h[idx]));
        EXPECT_TRUE(CEq(hn[idx], expi, 1e-12))
            << "idx="<<idx;
    }

    cudaFree(d);
}
