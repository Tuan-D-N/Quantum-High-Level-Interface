#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <span>
#include <optional>
#include <iomanip>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>

#include "../CuSparseControl/SparseDenseConvert.hpp"
#include "../CuSparseControl/ChebyshevMatrixExponential.hpp"
#include "../CuSparseControl/TaylorSparseMatrixExponential.hpp"

// ---- helpers ----
static inline cuDoubleComplex C(double r, double i=0.0) { return make_cuDoubleComplex(r,i); }

static inline bool CEq(cuDoubleComplex a, cuDoubleComplex b, double eps=1e-10) {
    return std::abs(cuCreal(a)-cuCreal(b)) <= eps && std::abs(cuCimag(a)-cuCimag(b)) <= eps;
}

#define CUSPARSE_ASSERT_OK(expr) do {                           \
    cusparseStatus_t _s = (expr);                                \
    ASSERT_EQ(_s, CUSPARSE_STATUS_SUCCESS) << "cuSPARSE error " << int(_s); \
} while(0)

// device mem helpers (void so ASSERT_* is fine)
static void uploadCSR(const std::vector<int>& hRow,
                      const std::vector<int>& hCol,
                      const std::vector<cuDoubleComplex>& hVal,
                      int** dRow, int** dCol, cuDoubleComplex** dVal)
{
    ASSERT_NE(dRow, nullptr); ASSERT_NE(dCol, nullptr); ASSERT_NE(dVal, nullptr);
    ASSERT_EQ(cudaMalloc((void**)dRow, sizeof(int)*hRow.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void**)dCol, sizeof(int)*hCol.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc((void**)dVal, sizeof(cuDoubleComplex)*hVal.size()), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(*dRow, hRow.data(), sizeof(int)*hRow.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(*dCol, hCol.data(), sizeof(int)*hCol.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(*dVal, hVal.data(), sizeof(cuDoubleComplex)*hVal.size(), cudaMemcpyHostToDevice), cudaSuccess);
}

static void uploadVec(const std::vector<cuDoubleComplex>& hVec, cuDoubleComplex** dVec) {
    ASSERT_NE(dVec, nullptr);
    ASSERT_EQ(cudaMalloc((void**)dVec, sizeof(cuDoubleComplex)*hVec.size()), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(*dVec, hVec.data(), sizeof(cuDoubleComplex)*hVec.size(), cudaMemcpyHostToDevice), cudaSuccess);
}

static void downloadVec(cuDoubleComplex* dVec, size_t n, std::vector<cuDoubleComplex>& out) {
    out.resize(n);
    ASSERT_EQ(cudaMemcpy(out.data(), dVec, sizeof(cuDoubleComplex)*n, cudaMemcpyDeviceToHost), cudaSuccess);
}

static void expectClose(const std::vector<cuDoubleComplex>& a,
                        const std::vector<cuDoubleComplex>& b,
                        double eps=1e-10)
{
    ASSERT_EQ(a.size(), b.size());
    for (size_t i=0;i<a.size();++i) {
        EXPECT_TRUE(CEq(a[i], b[i], eps))
            << std::setprecision(16)
            << "Mismatch at i="<<i
            << "\n  got      = ("<<cuCreal(a[i])<<", "<<cuCimag(a[i])<<")"
            << "\n  expected = ("<<cuCreal(b[i])<<", "<<cuCimag(b[i])<<")"
            << "\n  |diff|   = ("<<std::abs(cuCreal(a[i])-cuCreal(b[i]))<<", "
                               <<std::abs(cuCimag(a[i])-cuCimag(b[i]))<<")";
    }
}

// -------- Random generators --------

// random dense Hermitian (n x n), scaled so max row-sum <= scale
static std::vector<cuDoubleComplex> randomHermitianDense(int n, double scale, std::mt19937_64& g) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<cuDoubleComplex> A(n*n, C(0.0));

    // fill upper triangle then mirror
    for (int i=0;i<n;i++) {
        double r = u(g);
        A[i*n + i] = C(r, 0.0);
        for (int j=i+1;j<n;j++) {
            double ar = u(g), ai = u(g);
            A[i*n + j] = C(ar, ai);
            A[j*n + i] = C(ar, -ai); // conj
        }
    }
    // scale to keep it tame for Taylor(30)
    // use max row 1-norm
    double maxRow = 0.0;
    for (int i=0;i<n;i++) {
        double s = 0.0;
        for (int j=0;j<n;j++) {
            auto z = A[i*n + j];
            s += std::abs(cuCreal(z)) + std::abs(cuCimag(z));
        }
        maxRow = std::max(maxRow, s);
    }
    double fac = (maxRow > 0.0) ? (scale / maxRow) : 1.0;
    for (auto& z : A) {
        z = C(cuCreal(z)*fac, cuCimag(z)*fac);
    }
    return A;
}

// sparse-ify: zero small entries by magnitude threshold; keep diagonal
static void thresholdToCSR(std::vector<cuDoubleComplex>& dense, int n,
                           double tol,
                           std::vector<int>& row_ptr, std::vector<int>& col_ind,
                           std::vector<cuDoubleComplex>& vals)
{
    // zero small entries
    for (int i=0;i<n;i++) {
        for (int j=0;j<n;j++) {
            auto& z = dense[i*n + j];
            if (i==j) continue; // preserve diagonal
            double mag1 = std::abs(cuCreal(z)) + std::abs(cuCimag(z));
            if (mag1 < tol) z = C(0.0, 0.0);
        }
    }
    dense_to_csr(std::span<const cuDoubleComplex>(dense), n, row_ptr, col_ind, vals, /*tol=*/0.0);
    // guarantee at least one nonzero
    if (vals.empty()) {
        // put identity tiny
        row_ptr = {0};
        col_ind.clear(); vals.clear();
        int nnz=0;
        for (int i=0;i<n;i++) {
            row_ptr.push_back(++nnz);
            col_ind.push_back(i);
            vals.push_back(C(1e-6, 0.0));
        }
    }
}

static std::vector<cuDoubleComplex> randomState(size_t dim, std::mt19937_64& g) {
    std::normal_distribution<double> nrm(0.0,1.0);
    std::vector<cuDoubleComplex> v(dim);
    double norm2 = 0.0;
    for (auto& z : v) {
        double r=nrm(g), i=nrm(g);
        z = C(r,i);
        norm2 += r*r + i*i;
    }
    double inv = (norm2>0.0)? 1.0/std::sqrt(norm2) : 1.0;
    for (auto& z : v) {
        z = C(cuCreal(z)*inv, cuCimag(z)*inv);
    }
    return v;
}

// ==================== TESTS ====================

TEST(MatrixExponentialRandomCompare, VectorEvolution_RandomHermitian_SmallN_Order30) {
    // Compare exp(+iA) on a vector: Chebyshev (t=-1) vs Taylor with order=30
    constexpr int ORDER = 30;
    constexpr int NUM_TRIALS = 12;
    std::vector<int> sizes = {2, 4}; // small dense -> CSR
    std::mt19937_64 gen(12345);

    cusparseHandle_t handle;
    CUSPARSE_ASSERT_OK(cusparseCreate(&handle));

    for (int n : sizes) {
        for (int trial=0; trial<NUM_TRIALS; ++trial) {
            auto A_dense = randomHermitianDense(n, /*scale=*/0.9, gen);
            std::vector<int> row_ptr, col_ind;
            std::vector<cuDoubleComplex> vals;
            thresholdToCSR(A_dense, n, /*tol=*/0.15, row_ptr, col_ind, vals);
            int nnz = (int)vals.size();

            int *dRow=nullptr, *dCol=nullptr; cuDoubleComplex *dVal=nullptr;
            uploadCSR(row_ptr, col_ind, vals, &dRow, &dCol, &dVal);

            auto v0 = randomState(n, gen);
            cuDoubleComplex *d_v_cheb=nullptr, *d_v_tay=nullptr;
            uploadVec(v0, &d_v_cheb);
            uploadVec(v0, &d_v_tay);

            // Chebyshev, t=-1 → exp(+iA)
            ASSERT_EQ(expiAv_chebyshev_gamma_cusparse_host(
                handle, n, nnz, ORDER,
                row_ptr, col_ind, vals, d_v_cheb, /*t=*/-1.0), 0);

            // Taylor, exp(+iA)
            ASSERT_EQ(expiAv_taylor_cusparse(
                handle, n, nnz, ORDER, dRow, dCol, dVal, d_v_tay), 0);

            std::vector<cuDoubleComplex> h_cheb, h_tay;
            downloadVec(d_v_cheb, n, h_cheb);
            downloadVec(d_v_tay,  n, h_tay);

            // Allow modest tolerance; both should match closely
            expectClose(h_cheb, h_tay, 1e-10);

            cudaFree(dRow); cudaFree(dCol); cudaFree(dVal);
            cudaFree(d_v_cheb); cudaFree(d_v_tay);
        }
    }

    CUSPARSE_ASSERT_OK(cusparseDestroy(handle));
}

TEST(MatrixExponentialRandomCompare, Controlled_1TargetQubit_Random_Order30) {
    // nQubits=3, target={1}, control={2}. A is 2x2 random Hermitian.
    constexpr int ORDER = 30;
    constexpr int NUM_TRIALS = 10;
    const int n_target = 2; // one target qubit
    const int nQubits   = 3;
    const size_t dim = 1u << nQubits;
    std::mt19937_64 gen(987654321);

    cusparseHandle_t handle;
    CUSPARSE_ASSERT_OK(cusparseCreate(&handle));

    for (int trial=0; trial<NUM_TRIALS; ++trial) {
        auto A_dense = randomHermitianDense(n_target, /*scale=*/0.95, gen);
        std::vector<int> row_ptr, col_ind;
        std::vector<cuDoubleComplex> vals;
        thresholdToCSR(A_dense, n_target, /*tol=*/0.10, row_ptr, col_ind, vals);
        int nnz = (int)vals.size();

        int *dRow=nullptr, *dCol=nullptr; cuDoubleComplex *dVal=nullptr;
        uploadCSR(row_ptr, col_ind, vals, &dRow, &dCol, &dVal);

        auto psi0 = randomState(dim, gen);
        cuDoubleComplex *d_state_cheb=nullptr, *d_state_tay=nullptr;
        uploadVec(psi0, &d_state_cheb);
        uploadVec(psi0, &d_state_tay);

        const std::vector<int> target{1};
        const std::vector<int> control{2};

        // Chebyshev controlled: t=-1 → exp(+iA)
        ASSERT_EQ(applyControlledExpChebyshev_cusparse_host(
            handle, nQubits, dRow, dCol, dVal, d_state_cheb,
            target, control, nnz, ORDER, /*t=*/-1.0), 0);

        // Taylor controlled: exp(+iA)
        ASSERT_EQ(applyControlledExpTaylor_cusparse(
            handle, nQubits, dRow, dCol, dVal, d_state_tay,
            target, control, nnz, ORDER), 0);

        std::vector<cuDoubleComplex> h_cheb, h_tay;
        downloadVec(d_state_cheb, dim, h_cheb);
        downloadVec(d_state_tay,  dim, h_tay);
        expectClose(h_cheb, h_tay, 1e-10);

        cudaFree(dRow); cudaFree(dCol); cudaFree(dVal);
        cudaFree(d_state_cheb); cudaFree(d_state_tay);
    }

    CUSPARSE_ASSERT_OK(cusparseDestroy(handle));
}

TEST(MatrixExponentialRandomCompare, Controlled_2TargetQubits_Random_Order30) {
    // nQubits=3, target={0,1} (A is 4x4), control={2}
    constexpr int ORDER = 30;
    constexpr int NUM_TRIALS = 6;
    const int n_target = 4; // two target qubits => 4x4
    const int nQubits   = 3;
    const size_t dim = 1u << nQubits;
    std::mt19937_64 gen(20251013);

    cusparseHandle_t handle;
    CUSPARSE_ASSERT_OK(cusparseCreate(&handle));

    for (int trial=0; trial<NUM_TRIALS; ++trial) {
        auto A_dense = randomHermitianDense(n_target, /*scale=*/0.75, gen);
        std::vector<int> row_ptr, col_ind;
        std::vector<cuDoubleComplex> vals;
        thresholdToCSR(A_dense, n_target, /*tol=*/0.12, row_ptr, col_ind, vals);
        int nnz = (int)vals.size();

        int *dRow=nullptr, *dCol=nullptr; cuDoubleComplex *dVal=nullptr;
        uploadCSR(row_ptr, col_ind, vals, &dRow, &dCol, &dVal);

        auto psi0 = randomState(dim, gen);
        cuDoubleComplex *d_state_cheb=nullptr, *d_state_tay=nullptr;
        uploadVec(psi0, &d_state_cheb);
        uploadVec(psi0, &d_state_tay);

        const std::vector<int> target{0,1};
        const std::vector<int> control{2};

        ASSERT_EQ(applyControlledExpChebyshev_cusparse_host(
            handle, nQubits, dRow, dCol, dVal, d_state_cheb,
            target, control, nnz, ORDER, /*t=*/-1.0), 0);

        ASSERT_EQ(applyControlledExpTaylor_cusparse(
            handle, nQubits, dRow, dCol, dVal, d_state_tay,
            target, control, nnz, ORDER), 0);

        std::vector<cuDoubleComplex> h_cheb, h_tay;
        downloadVec(d_state_cheb, dim, h_cheb);
        downloadVec(d_state_tay,  dim, h_tay);
        expectClose(h_cheb, h_tay, 2e-10); // slightly looser for 4x4 targets

        cudaFree(dRow); cudaFree(dCol); cudaFree(dVal);
        cudaFree(d_state_cheb); cudaFree(d_state_tay);
    }

    CUSPARSE_ASSERT_OK(cusparseDestroy(handle));
}