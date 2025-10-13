#include <cuComplex.h> // cuDoubleComplex, make_cuDoubleComplex, cuCadd, cuCmul, ...
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <optional>
#include <span>
#include "ChebyshevMatrixExponentialCoeff.hpp"

// ============ small complex helpers ============
static cuDoubleComplex cplx(double re, double im = 0.0) { return make_cuDoubleComplex(re, im); }
static cuDoubleComplex cadd(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }
static cuDoubleComplex csub(cuDoubleComplex a, cuDoubleComplex b) { return cuCsub(a, b); }
static cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }
static cuDoubleComplex cscale(cuDoubleComplex a, double s) { return make_cuDoubleComplex(cuCreal(a) * s, cuCimag(a) * s); }
static cuDoubleComplex cconj(cuDoubleComplex a) { return make_cuDoubleComplex(cuCreal(a), -cuCimag(a)); }
static cuDoubleComplex cexp_minus_i(double theta) { return make_cuDoubleComplex(std::cos(theta), -std::sin(theta)); }

// ============ CSR helpers ============
// Compute ||A||_1 = max_j sum_i |a_ij| for a CSR matrix (row_ptr, col_ind, values)
static double csr_norm1(int n,
                         std::span<const int> row_ptr,
                         std::span<const int> col_ind,
                         std::span<const cuDoubleComplex> values)
{
    if ((int)row_ptr.size() != n + 1)
        throw std::invalid_argument("row_ptr.size() must be n+1");
    if (col_ind.size() != values.size())
        throw std::invalid_argument("col_ind and values size mismatch");

    std::vector<double> col_sum(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
        {
            int j = col_ind[p];
            if (j < 0 || j >= n)
                throw std::out_of_range("col_ind out of range");
            const cuDoubleComplex v = values[p];
            col_sum[j] += std::hypot(cuCreal(v), cuCimag(v));
        }
    }
    return *std::max_element(col_sum.begin(), col_sum.end());
}

// ============ Chebyshev T_k in monomial basis ============
// Build T_k(x): T_0=1, T_1=x, T_{k+1} = 2 x T_k - T_{k-1}
// Return coeffs[k][j] = coefficient of x^j in T_k(x), k=0..m.
static std::vector<std::vector<cuDoubleComplex>> cheb_T_polys_monomial(int m)
{
    std::vector<std::vector<cuDoubleComplex>> T;
    T.reserve(m + 1);
    T.push_back({cplx(1.0)}); // T0(x) = 1
    if (m == 0)
        return T;
    T.push_back({cplx(0.0), cplx(1.0)}); // T1(x) = x

    for (int k = 1; k < m; ++k)
    {
        const auto &Tk = T.back();
        const auto &Tkm1 = T[T.size() - 2];

        // two_x_Tk = 2 * x * Tk -> shift by 1 and scale by 2
        std::vector<cuDoubleComplex> two_x_Tk(Tk.size() + 1, cplx(0.0));
        for (size_t j = 0; j < Tk.size(); ++j)
            two_x_Tk[j + 1] = cscale(Tk[j], 2.0);

        // Padding to get 'a' and 'b' to be the same size
        const size_t L = std::max(two_x_Tk.size(), Tkm1.size());
        std::vector<cuDoubleComplex> a(L, cplx(0.0)), b(L, cplx(0.0));
        for (size_t j = 0; j < two_x_Tk.size(); ++j)
            a[j] = two_x_Tk[j];
        for (size_t j = 0; j < Tkm1.size(); ++j)
            b[j] = Tkm1[j];

        std::vector<cuDoubleComplex> Tnext(L, cplx(0.0));
        for (size_t j = 0; j < L; ++j)
            Tnext[j] = csub(a[j], b[j]); // 2xTk - T_{k-1}
        T.push_back(std::move(Tnext));
    }
    return T;
}

// ============ Bessel-J Chebyshev coefficients ============
// For e^{-i tb x} on [-1,1]:
//   c0 = 2 J0(tb),  ck = 2 (-i)^k Jk(tb), k>=1,
// then fold 1/2 into c0 (so synthesis is sum_k c_k T_k(x)).
static std::vector<cuDoubleComplex> chebyshev_unitary_coeffs(double tb, int m)
{
    std::vector<cuDoubleComplex> c(m + 1, cplx(0.0));
    c[0] = cplx(2.0 * std::cyl_bessel_j(0, tb));
    for (int k = 1; k <= m; ++k)
    {
        // (-i)^k = exp(-i*pi*k/2)
        const cuDoubleComplex pk = cexp_minus_i(M_PI * 0.5 * k);
        const double Jk = std::cyl_bessel_j(k, tb);
        c[k] = cscale(pk, 2.0 * Jk);
    }
    c[0] = cscale(c[0], 0.5); // fold 1/2
    return c;
}

// ============ PUBLIC: gamma coefficients (CSR, spectral-only) ============
// Spectral scaling only: Y = A / beta, where beta = spectral_radius (if provided)
// otherwise beta = ||A||_1 from CSR (safe upper bound).
//
// Returns gamma[0..m] such that  exp(-i t A) ≈ sum_{i=0}^m gamma[i] A^i.
//
// Inputs:
//   n              : dimension
//   row_ptr[n+1]  : CSR row pointer
//   col_ind[nnz]  : CSR column indices
//   values[nnz]   : CSR values (cuDoubleComplex)
//   t             : scalar
//   m             : degree
//   spectral_radius: optional beta (rho or known bound); if not set, uses ||A||_1
//
std::vector<cuDoubleComplex>
chebyshev_exp_gamma_spectral_csr(
    int n,
    std::span<const int> row_ptr,
    std::span<const int> col_ind,
    std::span<const cuDoubleComplex> values,
    double t,
    int m,
    std::optional<double> spectral_radius)
{
    if (n <= 0)
        throw std::invalid_argument("n must be > 0");
    if (m < 0)
        throw std::invalid_argument("m must be >= 0");
    if ((int)row_ptr.size() != n + 1)
        throw std::invalid_argument("row_ptr.size() must be n+1");
    if (col_ind.size() != values.size())
        throw std::invalid_argument("col_ind and values size mismatch");

    // 1) beta (spectral radius or safe bound)
    const double beta = spectral_radius.has_value() ? spectral_radius.value()
                                                    : csr_norm1(n, row_ptr, col_ind, values);
    if (beta <= 0.0)
    {
        // A ≈ 0 -> exp(-i t A) ≈ I -> gamma[0]=1, others 0
        std::vector<cuDoubleComplex> gamma(m + 1, cplx(0.0));
        gamma[0] = cplx(1.0);
        return gamma;
    }

    // 2) Chebyshev coefficients for e^{-i tb x}, x∈[-1,1]
    const double tb_abs = std::abs(t * beta);
    std::vector<cuDoubleComplex> ck = chebyshev_unitary_coeffs(tb_abs, m);
    if (t < 0.0)
    {
        // conjugate the series when we used |tb|
        for (auto &z : ck)
            z = cconj(z);
    }

    // 3) Chebyshev T_k in monomial basis
    std::vector<std::vector<cuDoubleComplex>> Tmono = cheb_T_polys_monomial(m);

    // 4) Polynomial in Y: polyY(x) = Σ_k c_k T_k(x)
    std::vector<cuDoubleComplex> polyY(m + 1, cplx(0.0));
    for (int k = 0; k <= m; ++k)
    {
        const auto &tk = Tmono[k];
        for (size_t j = 0; j < tk.size(); ++j)
        {
            polyY[j] = cadd(polyY[j], cmul(ck[k], tk[j]));
        }
    }

    // 5) Convert to gamma for A^i:
    //    Y = A / beta   (spectral-only => alpha = 0),
    //    so x^j = (A/beta)^j => only j=i contributes:
    //    gamma[i] = polyY[i] * beta^{-i}.
    std::vector<cuDoubleComplex> gamma(m + 1, cplx(0.0));
    for (int i = 0; i <= m; ++i)
    {
        const double beta_pow = std::pow(beta, i);
        gamma[i] = cscale(polyY[i], 1.0 / beta_pow);
    }

    return gamma;
}
