#include <gtest/gtest.h>
#include <vector>
#include <span>
#include <optional>
#include <cuComplex.h>
#include "../CuSparseControl/ChebyshevMatrixExponentialCoeff.hpp"
#include "../CuSparseControl/SparseDenseConvert.hpp"

// Simple helpers
static inline cuDoubleComplex C(double r, double i = 0.0)
{
    return make_cuDoubleComplex(r, i);
}
static inline bool CEq(cuDoubleComplex a, cuDoubleComplex b, double eps = 1e-12)
{
    return std::abs(cuCreal(a) - cuCreal(b)) <= eps && std::abs(cuCimag(a) - cuCimag(b)) <= eps;
}

TEST(ChebyshevExpGammaSpectral, ZeroMatrix_Order0)
{
    // A = 0_{2x2}, t arbitrary, m=0 (only c0 should appear)
    const int n = 2;
    std::vector<cuDoubleComplex> A(n * n, C(0));
    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A), n, row_ptr, col_ind, vals, 0.0);

    const double t = 1.2345;
    const int m = 0;
    auto coeffs = chebyshev_exp_gamma_spectral_csr(
        n, row_ptr, col_ind, vals, t, m, std::nullopt);

    ASSERT_EQ(coeffs.size(), static_cast<size_t>(m + 1));

    // TODO: Insert expected c0 for your definition.
    // For many standard normalizations, exp(-i t * 0) = 1  => c0 = 1, others 0 (but m=0 here).
    std::vector<cuDoubleComplex> expected = {
        C(1, 0)};

    for (int i = 0; i < expected.size(); ++i)
        EXPECT_TRUE(CEq(coeffs[i], expected[i], 1e-12));
}

TEST(ChebyshevExpGammaSpectral, ZeroMatrix_Order3)
{
    // A = 0_{3x3}, m=3 -> only c0 nonzero for typical scaling
    const int n = 3;
    std::vector<cuDoubleComplex> A(n * n, C(0));
    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A), n, row_ptr, col_ind, vals, 0.0);

    const double t = 0.75;
    const int m = 3;
    auto coeffs = chebyshev_exp_gamma_spectral_csr(
        n, row_ptr, col_ind, vals, t, m, std::nullopt);

    ASSERT_EQ(coeffs.size(), static_cast<size_t>(m + 1));

    std::vector<cuDoubleComplex> expected = {
        C(1, 0), C(0, 0), C(0, 0), C(0, 0)};
    for (int i = 0; i < expected.size(); ++i)
        EXPECT_TRUE(CEq(coeffs[i], expected[i], 1e-12));
    // TODO: Put expected coefficients c0..c3 here (many schemes give c0 = 1, c1..c3 = 0).
    // std::vector<cuDoubleComplex> expected = { C(1,0), C(0,0), C(0,0), C(0,0) };
    // for (int k = 0; k <= m; ++k) EXPECT_TRUE(CEq(coeffs[k], expected[k], 1e-12)) << "k="<<k;
}

TEST(ChebyshevExpGammaSpectral, DiagonalWithinMinus1To1_NoRadiusProvided)
{
    // A = diag(-1, 0, 1). Radius = 1 naturally. Good to test mapping without explicit radius.
    const int n = 3;
    std::vector<cuDoubleComplex> A = {
        C(-1), C(0), C(0),
        C(0), C(0), C(0),
        C(0), C(0), C(1)};
    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A), n, row_ptr, col_ind, vals, 0.0);

    const double t = 0.5;
    const int m = 6;
    auto coeffs = chebyshev_exp_gamma_spectral_csr(
        n, row_ptr, col_ind, vals, t, m, std::nullopt);

    ASSERT_EQ(coeffs.size(), static_cast<size_t>(m + 1));

    std::vector<cuDoubleComplex> expected = {
        C(0.99999999924783245, 0), C(0, -0.49999983158968481), C(-0.12499997592122218, 0), C(0, 0.020831985046352253), C(0.002604046194263342, 0), C(0, -0.00025771607172343926),
        C(-2.1508381623160645e-05, 0)};
    for (int i = 0; i < expected.size(); ++i)
        EXPECT_TRUE(CEq(coeffs[i], expected[i], 1e-12));
    // TODO: Insert your ground-truth coefficients for this setup (radius inferred).
    // std::vector<cuDoubleComplex> expected = { /* c0..c6 */ };
    // for (int k = 0; k <= m; ++k) EXPECT_TRUE(CEq(coeffs[k], expected[k], 1e-10)) << "k="<<k;
}

TEST(ChebyshevExpGammaSpectral, DiagonalScaled_WithRadiusProvided)
{
    // A = diag(-2, 0, 2). Provide spectral_radius = 2 explicitly to match scaling
    const int n = 3;
    std::vector<cuDoubleComplex> A = {
        C(-2), C(0), C(0),
        C(0), C(0), C(0),
        C(0), C(0), C(2)};
    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A), n, row_ptr, col_ind, vals, 0.0);

    const double t = 1.0;
    const int m = 8;
    const double rho = 2.0;

    auto coeffs = chebyshev_exp_gamma_spectral_csr(
        n, row_ptr, col_ind, vals, t, m, rho);

    ASSERT_EQ(coeffs.size(), static_cast<size_t>(m + 1));

    std::vector<cuDoubleComplex> expected = {
        C(0.99999949303580393, 0), C(0, -0.99997731348351626), C(-0.49999364142862818, 0), C(0, 0.16659061530191635), C(0.041653884684066902, 0), C(0, -0.0082642382799496053),
        C(-0.0013798653900933999, 0), C(0, 0.00017494407486827413), C(2.2179552287925885e-05, 0)};
    for (int i = 0; i < expected.size(); ++i)
        EXPECT_TRUE(CEq(coeffs[i], expected[i], 1e-12));
    // TODO: Insert your ground-truth coefficients c0..c8 for this scaling (rho=2.)
    // std::vector<cuDoubleComplex> expected = { /* c0..c8 */ };
    // for (int k = 0; k <= m; ++k) EXPECT_TRUE(CEq(coeffs[k], expected[k], 1e-10)) << "k="<<k;
}

TEST(ChebyshevExpGammaSpectral, RadiusProvidedMatchesInferred)
{
    // A in [-1,1], so inferred radius ~1. Provide rho=1 and expect identical coefficients.
    const int n = 2;
    std::vector<cuDoubleComplex> A = {
        C(0.25), C(0.0),
        C(0.0), C(-0.75)};
    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A), n, row_ptr, col_ind, vals, 0.0);

    const double t = 0.33;
    const int m = 7;

    auto coeffs_auto = chebyshev_exp_gamma_spectral_csr(
        n, row_ptr, col_ind, vals, t, m, std::nullopt);

    auto coeffs_rho1 = chebyshev_exp_gamma_spectral_csr(
        n, row_ptr, col_ind, vals, t, m, 1.0);

    ASSERT_EQ(coeffs_auto.size(), coeffs_rho1.size());
    for (int k = 0; k <= m; ++k)
    {
        EXPECT_TRUE(CEq(coeffs_auto[k], coeffs_rho1[k], 1e-8)) << "k=" << k;
    }

    std::vector<cuDoubleComplex> expected = {
        C(0.99999999999727585, 0), C(0, -0.32999999999955049), C(-0.054449999845020212, 0), C(0, 0.0059894999893434437), C(0.00049413237218208386, 0), C(0, -3.2612759288341123e-05),
        C(-1.789785135080624e-06, 0), C(0, 8.4398669966620385e-08)};
    for (int i = 0; i < expected.size(); ++i)
        EXPECT_TRUE(CEq(coeffs_auto[i], expected[i], 1e-12));
}

TEST(ChebyshevExpGammaSpectral, SmallHermitianOffDiagonal)
{
    // A = [[0, a], [a, 0]] with a=0.5 (eigs ±0.5). Nice symmetric case.
    const int n = 2;
    const double a = 0.5;
    std::vector<cuDoubleComplex> A = {
        C(0.0), C(a),
        C(a), C(0.0)};
    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A), n, row_ptr, col_ind, vals, 0.0);

    const double t = 0.8;
    const int m = 10;

    // Compare auto radius vs provided radius = 0.5
    auto coeffs_auto = chebyshev_exp_gamma_spectral_csr(
        n, row_ptr, col_ind, vals, t, m, std::nullopt);
    auto coeffs_rho = chebyshev_exp_gamma_spectral_csr(
        n, row_ptr, col_ind, vals, t, m, /*rho=*/0.5);

    ASSERT_EQ(coeffs_auto.size(), static_cast<size_t>(m + 1));
    ASSERT_EQ(coeffs_rho.size(), static_cast<size_t>(m + 1));
    for (int k = 0; k <= m; ++k)
    {
        EXPECT_TRUE(CEq(coeffs_auto[k], coeffs_rho[k], 1e-8)) << "k=" << k;
    }

    std::vector<cuDoubleComplex> expected = {
        C(0.99999999999999978, 0), C(0, -0.79999999999997762), C(-0.31999999999999507, 0), C(0, 0.085333333331532688), C(0.01706666666643742, 0), C(0, -0.0027306666263225317),
        C(-0.00036408888497584037, 0), C(0, 4.1609789757611177e-05), C(4.1609856782534564e-06, 0), C(0, -3.6839129225412614e-07), C(-2.9482027368081748e-08, 0)};
    for (int i = 0; i < expected.size(); ++i)
        EXPECT_TRUE(CEq(coeffs_auto[i], expected[i], 1e-12));
}

TEST(ChebyshevExpGammaSpectral, LargerOrderStability)
{
    // Random-ish sparse diagonal with entries inside [-1,1]
    const int n = 5;
    std::vector<cuDoubleComplex> A = {
        C(1.0), C(0), C(0), C(0), C(0),
        C(0), C(-0.6), C(0), C(0), C(0),
        C(0), C(0), C(0.2), C(0), C(0),
        C(0), C(0), C(0), C(0.0), C(0),
        C(0), C(0), C(0), C(0), C(-1.0)};
    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A), n, row_ptr, col_ind, vals, 0.0);

    const double t = 1.2;
    const int m = 24;

    auto coeffs = chebyshev_exp_gamma_spectral_csr(
        n, row_ptr, col_ind, vals, t, m, std::nullopt);

    // Basic sanity checks
    ASSERT_EQ(coeffs.size(), static_cast<size_t>(m + 1));
    // Coeff magnitudes should generally decay for smooth f; check not exploding.
    double max_abs = 0.0;
    for (auto c : coeffs)
    {
        double mag = std::hypot(cuCreal(c), cuCimag(c));
        max_abs = std::max(max_abs, mag);
    }
    EXPECT_LT(max_abs, 10.0); // Loose sanity bound (adjust if your normalization differs)

    std::vector<cuDoubleComplex> expected = {
        C(0.99999999999999989, 0), C(0, -1.2), C(-0.71999999999999986, 0), C(0, 0.28800000000000003), C(0.086399999999999935, 0), C(0, -0.020736000000000001),
        C(-0.0041471999999999985, 0), C(0, 0.000710948571428571), C(0.00010664228571428563, 0), C(0, -1.4218971428571457e-05), C(-1.7062765714285708e-06, 0), C(0, 1.8613926233766221e-07),
        C(1.8613926233766231e-08, 0), C(0, -1.718208575424521e-09), C(-1.4727502075067458e-10, 0), C(0, 1.1782001659939246e-11), C(8.8365012449716713e-13, 0), C(0, -6.23753027369587e-14),
        C(-4.1583535175979781e-15, 0), C(0, 2.6263268797988696e-16), C(1.5757962472949497e-17, 0), C(0, -9.0035080447848086e-19), C(-4.9110498797349105e-20, 0), C(0, 2.5243981753098537e-21),
        C(1.2629572821629027e-22, 0)};
    for (int i = 0; i < expected.size(); ++i)
        EXPECT_TRUE(CEq(coeffs[i], expected[i], 1e-12));
}
