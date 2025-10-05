#include <gtest/gtest.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <vector>
#include <complex>
#include <cmath>
#include "../CuSparseControl/SparseMatrixExponential.hpp"

static bool deviceNear(const cuDoubleComplex *d_vec,
                       const std::vector<std::complex<double>> &ref,
                       int n, double tol = 1e-8)
{
    std::vector<cuDoubleComplex> h(n);
    cudaMemcpy(h.data(), d_vec, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        if (std::abs(std::complex<double>(h[i].x, h[i].y) - ref[i]) > tol)
            return false;
    return true;
}

static bool deviceNear(const cuDoubleComplex *d_vec1,
                       const cuDoubleComplex *d_vec2,
                       int n,
                       double tol = 1e-8)
{
    std::vector<cuDoubleComplex> h1(n), h2(n);
    cudaMemcpy(h1.data(), d_vec1, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h2.data(), d_vec2, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i)
    {
        double dr = h1[i].x - h2[i].x;
        double di = h1[i].y - h2[i].y;
        double diff = std::sqrt(dr * dr + di * di);
        if (diff > tol)
        {
            std::cerr << "Mismatch at index " << i << "\n"
                      << "  Actual:   (" << h1[i].x << ", " << h1[i].y << "i)\n"
                      << "  Expected: (" << h2[i].x << ", " << h2[i].y << "i)\n"
                      << "  Difference magnitude: " << diff << "\n"
                      << "  Tolerance: " << tol << std::endl;
            return false;
        }
    }
    return true;
}

static void runPauliTest(const std::vector<cuDoubleComplex> &A,
                         const std::vector<std::complex<double>> &vin,
                         const std::vector<std::complex<double>> &vexp,
                         int n, int order = 25)
{
    cusparseHandle_t h;
    cusparseCreate(&h);
    int nnz = n * n;
    std::vector<int> row(n + 1), col(nnz);
    for (int i = 0; i <= n; ++i)
        row[i] = i * n;
    for (int i = 0; i < nnz; ++i)
        col[i] = i % n;
    int *drow, *dcol;
    cuDoubleComplex *dval, *dvec;
    cudaMalloc(&drow, (n + 1) * sizeof(int));
    cudaMalloc(&dcol, nnz * sizeof(int));
    cudaMalloc(&dval, nnz * sizeof(cuDoubleComplex));
    cudaMemcpy(drow, row.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dcol, col.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dval, A.data(), nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    std::vector<cuDoubleComplex> hv(n);
    for (int i = 0; i < n; ++i)
        hv[i] = make_cuDoubleComplex(vin[i].real(), vin[i].imag());
    cudaMalloc(&dvec, n * sizeof(cuDoubleComplex));
    cudaMemcpy(dvec, hv.data(), n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    expiAv_taylor_cusparse(h, n, nnz, order, drow, dcol, dval, dvec);
    EXPECT_TRUE(deviceNear(dvec, vexp, n));
    cudaFree(drow);
    cudaFree(dcol);
    cudaFree(dval);
    cudaFree(dvec);
    cusparseDestroy(h);
}

TEST(ExpiPauliSparse, SigmaX)
{
    double theta = M_PI / 4;
    std::vector<cuDoubleComplex> pauli_x = {
        {0, 0}, {theta, 0}, {theta, 0}, {0, 0}};
    runPauliTest(pauli_x, {{1, 0}, {0, 0}},
                 {{cos(theta), 0}, {0, sin(theta)}}, 2);
}

TEST(ExpiPauliSparse, SigmaZ)
{
    double theta = M_PI / 3, s = 1 / std::sqrt(2.0);
    std::vector<cuDoubleComplex> pauli_z = {
        {theta, 0}, {0, 0}, {0, 0}, {-theta, 0}};
    runPauliTest(pauli_z, {{s, 0}, {s, 0}},
                 {{s * cos(theta), s * sin(theta)}, {s * cos(-theta), s * sin(-theta)}}, 2);
}

// Controlled 2-qubit test
TEST(ExpiPauliSparse, ControlledX)
{
    cusparseHandle_t h;
    cusparseCreate(&h);
    double theta = M_PI / 4;
    int nQ = 2, d = 2, nnz = 4, order = 25;
    std::vector<int> row = {0, 2, 4}, col = {0, 1, 0, 1};
    std::vector<cuDoubleComplex> val = {
        {0, 0}, {theta, 0}, {theta, 0}, {0, 0}};
    int *drow, *dcol;
    cuDoubleComplex *dval, *dstate;
    cudaMalloc(&drow, (d + 1) * sizeof(int));
    cudaMalloc(&dcol, nnz * sizeof(int));
    cudaMalloc(&dval, nnz * sizeof(cuDoubleComplex));
    cudaMemcpy(drow, row.data(), (d + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dcol, col.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dval, val.data(), nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    std::vector<cuDoubleComplex> hstate(4, {0, 0});
    hstate[2] = {1, 0};
    cudaMalloc(&dstate, 4 * sizeof(cuDoubleComplex));
    cudaMemcpy(dstate, hstate.data(), 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    std::vector<int> targ = {0}, ctrl = {1};
    applyControlledExpTaylor_cusparse(h, nQ, drow, dcol, dval, dstate, targ, ctrl, nnz, order);
    std::vector<std::complex<double>> ref(4, {0, 0});
    ref[2] = {cos(theta), 0};
    ref[3] = {0, sin(theta)};
    EXPECT_TRUE(deviceNear(dstate, ref, 4));
    cudaFree(drow);
    cudaFree(dcol);
    cudaFree(dval);
    cudaFree(dstate);
    cusparseDestroy(h);
}

// ======================================================
// σx on first qubit
// ======================================================
TEST(ExpiPauliSparse, SigmaX_FirstQubit)
{
    double theta = M_PI / 6;
    // σx ⊗ I (4×4)
    std::vector<cuDoubleComplex> A = {
        {0, 0}, {theta, 0}, {0, 0}, {0, 0}, {theta, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {theta, 0}, {0, 0}, {0, 0}, {theta, 0}, {0, 0}};
    runPauliTest(A,
                 {{1, 0}, {0, 0}, {0, 0}, {0, 0}}, // |00>
                 {{cos(theta), 0}, {0, sin(theta)}, {0, 0}, {0, 0}},
                 4, 30);
}

// ======================================================
// σx on second qubit
// ======================================================
TEST(ExpiPauliSparse, SigmaX_SecondQubit)
{
    double theta = M_PI / 5;
    // I ⊗ σx (4×4)
    std::vector<cuDoubleComplex> A = {
        {0, 0}, {theta, 0}, {0, 0}, {0, 0}, {theta, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {theta, 0}, {0, 0}, {0, 0}, {theta, 0}, {0, 0}};
    runPauliTest(A,
                 {{1, 0}, {0, 0}, {0, 0}, {0, 0}}, // |00>
                 {{cos(theta), 0}, {0, sin(theta)}, {0, 0}, {0, 0}},
                 4, 30);
}

// ======================================================
// σx⊗σx — simultaneous flip on both qubits
// ======================================================
TEST(ExpiPauliSparse, SigmaX_Tensor_SigmaX)
{
    double theta = M_PI / 4;
    // σx⊗σx (4×4)
    std::vector<cuDoubleComplex> A = {
        {0, 0}, {0, 0}, {0, 0}, {theta, 0}, {0, 0}, {0, 0}, {theta, 0}, {0, 0}, {0, 0}, {theta, 0}, {0, 0}, {0, 0}, {theta, 0}, {0, 0}, {0, 0}, {0, 0}};
    runPauliTest(A,
                 {{1, 0}, {0, 0}, {0, 0}, {0, 0}}, // |00>
                 {{cos(theta), 0}, {0, 0}, {0, 0}, {0, sin(theta)}},
                 4, 30);
}

// ======================================================
// σz⊗σz — phase rotation
// ======================================================
TEST(ExpiPauliSparse, SigmaZ_Tensor_SigmaZ)
{
    double theta = M_PI / 3;
    // σz⊗σz = diag(1, -1, -1, 1)
    std::vector<cuDoubleComplex> A = {
        {theta, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {-theta, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {-theta, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {theta, 0}};
    double c = std::cos(theta), s = std::sin(theta);
    runPauliTest(A,
                 {{1 / std::sqrt(2.0), 0}, {1 / std::sqrt(2.0), 0}, {0, 0}, {0, 0}}, // (|00>+|01>)/√2
                 {{c / sqrt(2.0), s / sqrt(2.0)}, {c / sqrt(2.0), -s / sqrt(2.0)}, {0, 0}, {0, 0}},
                 4, 30);
}

TEST(ExpiPauliSparse, Controlled_NoControl_Equals_Uncontrolled4by4)
{
    double theta = M_PI / 4;
    int nQubits = 2;
    int d = 4;
    int nnz = d * d;
    int order = 30;

    // σx ⊗ I  (acts only on first qubit)
    std::vector<cuDoubleComplex> A = {
        {0, 0}, {theta, 0}, {0, 0}, {0, 0}, {theta, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {theta, 0}, {0, 0}, {0, 0}, {theta, 0}, {0, 0}};

    // Initial |00>
    std::vector<cuDoubleComplex> h_state(d, {0, 0});
    h_state[0] = {1, 0};

    // Device setup
    std::vector<int> row(d + 1), col(nnz);
    for (int i = 0; i <= d; ++i)
        row[i] = i * d;
    for (int i = 0; i < nnz; ++i)
        col[i] = i % d;

    int *d_row, *d_col;
    cuDoubleComplex *d_val, *d_state1, *d_state2;
    cudaMalloc(&d_row, (d + 1) * sizeof(int));
    cudaMalloc(&d_col, nnz * sizeof(int));
    cudaMalloc(&d_val, nnz * sizeof(cuDoubleComplex));
    cudaMalloc(&d_state1, d * sizeof(cuDoubleComplex));
    cudaMalloc(&d_state2, d * sizeof(cuDoubleComplex));
    cudaMemcpy(d_row, row.data(), (d + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, A.data(), nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_state1, h_state.data(), d * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_state2, h_state.data(), d * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Run uncontrolled version
    expiAv_taylor_cusparse(handle, d, nnz, order, d_row, d_col, d_val, d_state1);

    // Run controlled version (no controls)
    std::vector<int> targetQubits = {0, 1};
    std::vector<int> controlQubits; // empty
    applyControlledExpTaylor_cusparse(handle, nQubits,
                                      d_row, d_col, d_val, d_state2,
                                      targetQubits, controlQubits,
                                      nnz, order);

    // Compare
    EXPECT_TRUE(deviceNear(d_state1, d_state2, d));

    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_val);
    cudaFree(d_state1);
    cudaFree(d_state2);
    cusparseDestroy(handle);
}
