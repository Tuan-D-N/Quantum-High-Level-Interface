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
static inline cuDoubleComplex C(double r, double i = 0.0) { return make_cuDoubleComplex(r, i); }

static inline bool CEq(cuDoubleComplex a, cuDoubleComplex b, double eps = 1e-12)
{
    return std::abs(cuCreal(a) - cuCreal(b)) <= eps && std::abs(cuCimag(a) - cuCimag(b)) <= eps;
}

#define CUDA_OK(expr)                                         \
    do                                                        \
    {                                                         \
        cudaError_t _e = (expr);                              \
        ASSERT_EQ(_e, cudaSuccess) << cudaGetErrorString(_e); \
    } while (0)

#define CUSPARSE_OK(expr)                                                            \
    do                                                                               \
    {                                                                                \
        cusparseStatus_t _s = (expr);                                                \
        ASSERT_EQ(_s, CUSPARSE_STATUS_SUCCESS) << "cuSPARSE error code " << int(_s); \
    } while (0)

static void uploadCSR(const std::vector<int> &hRow,
                      const std::vector<int> &hCol,
                      const std::vector<cuDoubleComplex> &hVal,
                      int **dRow, int **dCol, cuDoubleComplex **dVal)
{
    CUDA_OK(cudaMalloc((void **)dRow, sizeof(int) * hRow.size()));
    CUDA_OK(cudaMalloc((void **)dCol, sizeof(int) * hCol.size()));
    CUDA_OK(cudaMalloc((void **)dVal, sizeof(cuDoubleComplex) * hVal.size()));
    CUDA_OK(cudaMemcpy(*dRow, hRow.data(), sizeof(int) * hRow.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(*dCol, hCol.data(), sizeof(int) * hCol.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(*dVal, hVal.data(), sizeof(cuDoubleComplex) * hVal.size(), cudaMemcpyHostToDevice));
}

static void uploadVec(const std::vector<cuDoubleComplex> &hVec, cuDoubleComplex **dVec)
{
    CUDA_OK(cudaMalloc((void **)dVec, sizeof(cuDoubleComplex) * hVec.size()));
    CUDA_OK(cudaMemcpy(*dVec, hVec.data(), sizeof(cuDoubleComplex) * hVec.size(), cudaMemcpyHostToDevice));
}

static std::vector<cuDoubleComplex> downloadVec(cuDoubleComplex* dVec, size_t n) {
    std::vector<cuDoubleComplex> h(n);
    cudaError_t e = cudaMemcpy(h.data(), dVec, sizeof(cuDoubleComplex)*n, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        ADD_FAILURE() << "cudaMemcpy D2H failed: " << cudaGetErrorString(e);
        return {}; // return empty to signal failure
    }
    return h;
}


static void expectClose(const std::vector<cuDoubleComplex> &a,
                        const std::vector<cuDoubleComplex> &b,
                        double eps = 1e-10)
{
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i)
    {
        EXPECT_TRUE(CEq(a[i], b[i], eps))
            << std::setprecision(16)
            << "Mismatch at i=" << i
            << "\n  got      = (" << cuCreal(a[i]) << ", " << cuCimag(a[i]) << ")"
            << "\n  expected = (" << cuCreal(b[i]) << ", " << cuCimag(b[i]) << ")"
            << "\n  |diff|   = (" << std::abs(cuCreal(a[i]) - cuCreal(b[i])) << ", "
            << std::abs(cuCimag(a[i]) - cuCimag(b[i])) << ")";
    }
}

// Build Pauli X (sigma_x) as dense n=2
static std::vector<cuDoubleComplex> pauliX_dense()
{
    return {C(0), C(1),
            C(1), C(0)};
}

// Build Pauli Z (sigma_z) as dense n=2
static std::vector<cuDoubleComplex> pauliZ_dense()
{
    return {C(1), C(0),
            C(0), C(-1)};
}

// Random complex vector
static std::vector<cuDoubleComplex> randomVec(size_t n, unsigned seed = 42)
{
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<cuDoubleComplex> v(n);
    for (auto &z : v)
        z = C(dist(gen), dist(gen));
    return v;
}

// ============== TESTS ==============

TEST(ExpiAvCompare, ChebyshevVsTaylor_Vector_PauliX_Order12)
{
    // Compare exp(+iA) via:
    //  - Chebyshev host (set t = -1 to get exp(+iA))
    //  - Taylor kernel
    const int n = 2;
    auto A_dense = pauliX_dense();

    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A_dense), n, row_ptr, col_ind, vals, 0.0);
    const int nnz = (int)vals.size();

    int *dRow = nullptr, *dCol = nullptr;
    cuDoubleComplex *dVal = nullptr;
    uploadCSR(row_ptr, col_ind, vals, &dRow, &dCol, &dVal);

    auto v0 = randomVec(n);
    cuDoubleComplex *d_v_cheb = nullptr, *d_v_tay = nullptr;
    uploadVec(v0, &d_v_cheb);
    uploadVec(v0, &d_v_tay);

    cusparseHandle_t handle;
    CUSPARSE_OK(cusparseCreate(&handle));

    const int order = 12;

    // Chebyshev: t = -1 → exp(+i A)
    {
        int rc = expiAv_chebyshev_gamma_cusparse_host(
            handle, n, nnz, order,
            std::span<const int>(row_ptr), std::span<const int>(col_ind), std::span<const cuDoubleComplex>(vals),
            d_v_cheb, /*t=*/-1.0);
        ASSERT_EQ(rc, 0);
    }
    // Taylor: exp(+iA)
    {
        int rc = expiAv_taylor_cusparse(handle, n, nnz, order, dRow, dCol, dVal, d_v_tay);
        ASSERT_EQ(rc, 0);
    }

    auto h_cheb = downloadVec(d_v_cheb, n);
    auto h_tay = downloadVec(d_v_tay, n);

    expectClose(h_cheb, h_tay, 1e-10);

    CUSPARSE_OK(cusparseDestroy(handle));
    CUDA_OK(cudaFree(dRow));
    CUDA_OK(cudaFree(dCol));
    CUDA_OK(cudaFree(dVal));
    CUDA_OK(cudaFree(d_v_cheb));
    CUDA_OK(cudaFree(d_v_tay));
}

TEST(ExpiAvCompare, ChebyshevVsTaylor_Vector_PauliZ_Order10)
{
    const int n = 2;
    auto A_dense = pauliZ_dense();

    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A_dense), n, row_ptr, col_ind, vals, 0.0);
    const int nnz = (int)vals.size();

    int *dRow = nullptr, *dCol = nullptr;
    cuDoubleComplex *dVal = nullptr;
    uploadCSR(row_ptr, col_ind, vals, &dRow, &dCol, &dVal);

    auto v0 = randomVec(n, 7);
    cuDoubleComplex *d_v_cheb = nullptr, *d_v_tay = nullptr;
    uploadVec(v0, &d_v_cheb);
    uploadVec(v0, &d_v_tay);

    cusparseHandle_t handle;
    CUSPARSE_OK(cusparseCreate(&handle));

    const int order = 10;
    int rc1 = expiAv_chebyshev_gamma_cusparse_host(handle, n, nnz, order,
                                                   row_ptr, col_ind, vals, d_v_cheb, /*t=*/-1.0);
    int rc2 = expiAv_taylor_cusparse(handle, n, nnz, order,
                                     dRow, dCol, dVal, d_v_tay);
    ASSERT_EQ(rc1, 0);
    ASSERT_EQ(rc2, 0);

    auto h_cheb = downloadVec(d_v_cheb, n);
    auto h_tay = downloadVec(d_v_tay, n);
    expectClose(h_cheb, h_tay, 1e-11);

    CUSPARSE_OK(cusparseDestroy(handle));
    CUDA_OK(cudaFree(dRow));
    CUDA_OK(cudaFree(dCol));
    CUDA_OK(cudaFree(dVal));
    CUDA_OK(cudaFree(d_v_cheb));
    CUDA_OK(cudaFree(d_v_tay));
}

TEST(ExpiAvCompare, Order0Identity_Vector)
{
    // order=0 should be identity for both methods
    const int n = 2;
    auto A_dense = pauliX_dense();

    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A_dense), n, row_ptr, col_ind, vals, 0.0);
    const int nnz = (int)vals.size();

    int *dRow = nullptr, *dCol = nullptr;
    cuDoubleComplex *dVal = nullptr;
    uploadCSR(row_ptr, col_ind, vals, &dRow, &dCol, &dVal);

    auto v0 = randomVec(n, 99);
    cuDoubleComplex *d_v_cheb = nullptr, *d_v_tay = nullptr;
    uploadVec(v0, &d_v_cheb);
    uploadVec(v0, &d_v_tay);

    cusparseHandle_t handle;
    CUSPARSE_OK(cusparseCreate(&handle));

    const int order = 0;
    ASSERT_EQ(expiAv_chebyshev_gamma_cusparse_host(handle, n, nnz, order,
                                                   row_ptr, col_ind, vals, d_v_cheb, /*t=*/-1.0),
              0);
    ASSERT_EQ(expiAv_taylor_cusparse(handle, n, nnz, order,
                                     dRow, dCol, dVal, d_v_tay),
              0);

    auto h_cheb = downloadVec(d_v_cheb, n);
    auto h_tay = downloadVec(d_v_tay, n);

    // both should equal v0
    expectClose(h_cheb, v0, 1e-15);
    expectClose(h_tay, v0, 1e-15);
    expectClose(h_cheb, h_tay, 1e-15);

    CUSPARSE_OK(cusparseDestroy(handle));
    CUDA_OK(cudaFree(dRow));
    CUDA_OK(cudaFree(dCol));
    CUDA_OK(cudaFree(dVal));
    CUDA_OK(cudaFree(d_v_cheb));
    CUDA_OK(cudaFree(d_v_tay));
}

TEST(ControlledCompare, ChebyshevVsTaylor_2Qubits_Target0_Control1_PauliZ)
{
    // nQubits=2, target={0}, control={1}. State dimension = 4.
    // Both functions should implement C(exp(+iA)) with t=-1 for Chebyshev; Taylor is inherently exp(+iA).
    const int n_target = 2;        // A is 2x2 (acts on one target qubit)
    auto A_dense = pauliZ_dense(); // Hermitian

    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A_dense), n_target, row_ptr, col_ind, vals, 0.0);
    const int nnz = (int)vals.size();

    int *dRow = nullptr, *dCol = nullptr;
    cuDoubleComplex *dVal = nullptr;
    uploadCSR(row_ptr, col_ind, vals, &dRow, &dCol, &dVal);

    const int nQubits = 2;
    const size_t dim = 1u << nQubits;

    // Create a state with mixed amplitudes so the control branch actually matters:
    std::vector<cuDoubleComplex> psi0 = {
        C(0.6, -0.1), // |00>
        C(-0.2, 0.4), // |01>
        C(0.1, 0.3),  // |10>   <-- controlled branch active (control=1)
        C(-0.5, 0.2)  // |11>   <-- controlled branch active
    };

    cuDoubleComplex *d_state_cheb = nullptr, *d_state_tay = nullptr;
    uploadVec(psi0, &d_state_cheb);
    uploadVec(psi0, &d_state_tay);

    cusparseHandle_t handle;
    CUSPARSE_OK(cusparseCreate(&handle));

    const int order = 12;
    const std::vector<int> target{0};
    const std::vector<int> control{1};

    // Chebyshev controlled: t=-1 → exp(+iA)
    {
        int rc = applyControlledExpChebyshev_cusparse_host(
            handle, nQubits, dRow, dCol, dVal, d_state_cheb,
            target, control, nnz, order, /*t=*/-1.0);
        ASSERT_EQ(rc, 0);
    }
    // Taylor controlled: exp(+iA)
    {
        int rc = applyControlledExpTaylor_cusparse(
            handle, nQubits, dRow, dCol, dVal, d_state_tay,
            target, control, nnz, order);
        ASSERT_EQ(rc, 0);
    }

    auto h_cheb = downloadVec(d_state_cheb, dim);
    auto h_tay = downloadVec(d_state_tay, dim);

    expectClose(h_cheb, h_tay, 1e-10);

    CUSPARSE_OK(cusparseDestroy(handle));
    CUDA_OK(cudaFree(dRow));
    CUDA_OK(cudaFree(dCol));
    CUDA_OK(cudaFree(dVal));
    CUDA_OK(cudaFree(d_state_cheb));
    CUDA_OK(cudaFree(d_state_tay));
}

TEST(ControlledCompare, Order0Identity_OnControlled)
{
    // With order=0, both controlled versions should act as identity (no change to state).
    const int n_target = 2;
    auto A_dense = pauliX_dense();

    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A_dense), n_target, row_ptr, col_ind, vals, 0.0);
    const int nnz = (int)vals.size();

    int *dRow = nullptr, *dCol = nullptr;
    cuDoubleComplex *dVal = nullptr;
    uploadCSR(row_ptr, col_ind, vals, &dRow, &dCol, &dVal);

    const int nQubits = 2;
    const size_t dim = 1u << nQubits;

    std::vector<cuDoubleComplex> psi0 = randomVec(dim, 2025);
    cuDoubleComplex *d_state_cheb = nullptr, *d_state_tay = nullptr;
    uploadVec(psi0, &d_state_cheb);
    uploadVec(psi0, &d_state_tay);

    cusparseHandle_t handle;
    CUSPARSE_OK(cusparseCreate(&handle));

    const int order = 0;
    const std::vector<int> target{1};  // act on qubit 1
    const std::vector<int> control{0}; // controlled by qubit 0 == |1>

    ASSERT_EQ(applyControlledExpChebyshev_cusparse_host(
                  handle, nQubits, dRow, dCol, dVal, d_state_cheb,
                  target, control, nnz, order, /*t=*/-1.0),
              0);
    ASSERT_EQ(applyControlledExpTaylor_cusparse(
                  handle, nQubits, dRow, dCol, dVal, d_state_tay,
                  target, control, nnz, order),
              0);

    auto h_cheb = downloadVec(d_state_cheb, dim);
    auto h_tay = downloadVec(d_state_tay, dim);

    expectClose(h_cheb, psi0, 1e-15);
    expectClose(h_tay, psi0, 1e-15);
    expectClose(h_cheb, h_tay, 1e-15);

    CUSPARSE_OK(cusparseDestroy(handle));
    CUDA_OK(cudaFree(dRow));
    CUDA_OK(cudaFree(dCol));
    CUDA_OK(cudaFree(dVal));
    CUDA_OK(cudaFree(d_state_cheb));
    CUDA_OK(cudaFree(d_state_tay));
}
