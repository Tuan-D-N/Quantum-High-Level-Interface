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
    double θ = M_PI / 4;
    std::vector<cuDoubleComplex> σx = {
        {0, 0}, {θ, 0}, {θ, 0}, {0, 0}};
    runPauliTest(σx, {{1, 0}, {0, 0}},
                 {{cos(θ), 0}, {0, sin(θ)}}, 2);
}

TEST(ExpiPauliSparse, SigmaZ)
{
    double θ = M_PI / 3, s = 1 / std::sqrt(2.0);
    std::vector<cuDoubleComplex> σz = {
        {θ, 0}, {0, 0}, {0, 0}, {-θ, 0}};
    runPauliTest(σz, {{s, 0}, {s, 0}},
                 {{s * cos(θ), s * sin(θ)}, {s * cos(-θ), s * sin(-θ)}}, 2);
}

// Controlled 2-qubit test
TEST(ExpiPauliSparse, ControlledX)
{
    cusparseHandle_t h;
    cusparseCreate(&h);
    double θ = M_PI / 4;
    int nQ = 2, d = 2, nnz = 4, order = 25;
    std::vector<int> row = {0, 2, 4}, col = {0, 1, 0, 1};
    std::vector<cuDoubleComplex> val = {
        {0, 0}, {θ, 0}, {θ, 0}, {0, 0}};
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
    ref[2] = {cos(θ), 0};
    ref[3] = {0, sin(θ)};
    EXPECT_TRUE(deviceNear(dstate, ref, 4));
    cudaFree(drow);
    cudaFree(dcol);
    cudaFree(dval);
    cudaFree(dstate);
    cusparseDestroy(h);
}
