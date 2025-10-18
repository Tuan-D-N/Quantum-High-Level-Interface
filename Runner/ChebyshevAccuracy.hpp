#include <custatevec.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>
#include "../CuQuantumControl/StateObject.hpp"
#include "../functionality/SquareNorm.hpp"
#include "../CuSparseControl/SparseDenseConvert.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>


// ---- Prototypes you provided ----
int expiAv_taylor_cusparse(
    cusparseHandle_t handle,
    int n, int nnz, int order,
    const int *d_csrRowPtr,
    const int *d_csrColInd,
    const cuDoubleComplex *d_csrVal,
    cuDoubleComplex *d_v_in_out /* assuming in/out */
);

int expiAv_chebyshev_gamma_cusparse_host(
    cusparseHandle_t handle,
    int n, int nnz, int order,
    std::span<const int> h_csrRowPtr,
    std::span<const int> h_csrColInd,
    std::span<const cuDoubleComplex> h_csrVal,
    cuDoubleComplex *d_v_in_out,
    const double t /* +1: exp(-iA), -1: exp(+iA) */
);

// ---- Build 1D Laplacian (Hermitian SPD) on n points: A(i,i)=2, A(i,i±1)=-1 ----
static void build_tridiag_laplacian(int n,
                                    std::vector<int>& h_rowPtr,
                                    std::vector<int>& h_colInd,
                                    std::vector<cuDoubleComplex>& h_vals)
{
    h_rowPtr.resize(n+1);
    std::vector<int> cols;
    std::vector<cuDoubleComplex> vals;
    cols.reserve(3*n);
    vals.reserve(3*n);

    int nnz = 0;
    for(int i=0;i<n;++i){
        h_rowPtr[i] = nnz;
        // left
        if(i-1>=0){ cols.push_back(i-1); vals.push_back(make_cuDoubleComplex(-1.0, 0.0)); ++nnz; }
        // center
        cols.push_back(i); vals.push_back(make_cuDoubleComplex(2.0, 0.0)); ++nnz;
        // right
        if(i+1<n){ cols.push_back(i+1); vals.push_back(make_cuDoubleComplex(-1.0, 0.0)); ++nnz; }
    }
    h_rowPtr[n] = nnz;
    h_colInd = std::move(cols);
    h_vals   = std::move(vals);
}

// ---- Host L2 norm and relative error ----
static double rel_error_l2(const std::vector<cuDoubleComplex>& y,
                           const std::vector<cuDoubleComplex>& yref)
{
    assert(y.size()==yref.size());
    long long n = (long long)y.size();
    long double num2=0.0L, den2=0.0L;
    for(long long i=0;i<n;++i){
        double dx = cuCreal(y[i]) - cuCreal(yref[i]);
        double dy = cuCimag(y[i]) - cuCimag(yref[i]);
        num2 += (long double)(dx*dx + dy*dy);
        double rx = cuCreal(yref[i]);
        double ry = cuCimag(yref[i]);
        den2 += (long double)(rx*rx + ry*ry);
    }
    if (den2 == 0.0L) return std::sqrt((double)num2);
    return std::sqrt((double)num2) / std::sqrt((double)den2);
}

// ---- Main driver ----
// --- randomize values on the fixed tri-diagonal pattern (SPD) ---
static void fill_tridiag_values_random_spd(
    int n,
    const std::vector<int>& h_rowPtr,
    const std::vector<int>& h_colInd,
    std::vector<cuDoubleComplex>& h_vals,
    std::mt19937_64& rng,
    double wmin = 0.1, double wmax = 1.0, double gamma = 1.0)
{
    std::uniform_real_distribution<double> U(wmin, wmax);
    h_vals.assign(h_colInd.size(), make_cuDoubleComplex(0.0, 0.0));

    // off-diagonals negative, diagonal = sum|off| + gamma  (strictly diag. dominant)
    std::vector<double> diag_accum(n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int k = h_rowPtr[i]; k < h_rowPtr[i + 1]; ++k) {
            int j = h_colInd[k];
            if (j == i) continue;
            // In tri-diagonal pattern, |i-j| is 1
            // Draw a negative real weight in [-wmax, -wmin]
            double w = -U(rng);
            h_vals[k] = make_cuDoubleComplex(w, 0.0);
            diag_accum[i] += std::abs(w);
        }
    }
    // set diagonals
    for (int i = 0; i < n; ++i) {
        for (int k = h_rowPtr[i]; k < h_rowPtr[i + 1]; ++k) {
            if (h_colInd[k] == i) {
                h_vals[k] = make_cuDoubleComplex(diag_accum[i] + gamma, 0.0);
            }
        }
    }
}

// ---- Main driver with trials & stats ----
int runCheby()
{
    // --- Settings ---
    const int n            = 256;  // matrix size
    const int order_min    = 2;
    const int order_max    = 64;
    const int order_step   = 2;
    const int order_ref    = 512;  // high-order Chebyshev reference
    const int trials       = 20;   // <-- number of trials
    const unsigned seed    = 1234u;

    // --- cuSPARSE handle ---
    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // --- Build fixed CSR pattern (tri-diagonal) once ---
    std::vector<int> h_rowPtr, h_colInd;
    std::vector<cuDoubleComplex> h_vals;
    build_tridiag_laplacian(n, h_rowPtr, h_colInd, h_vals); // initializes with {2,-1,-1}
    const int nnz = (int)h_colInd.size();

    // --- Device CSR (structure fixed; values change per-trial) ---
    int *d_rowPtr = nullptr, *d_colInd = nullptr;
    cuDoubleComplex *d_vals = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_rowPtr, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_colInd, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals,   nnz * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMemcpy(d_rowPtr, h_rowPtr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, h_colInd.data(), nnz * sizeof(int),     cudaMemcpyHostToDevice));

    // --- Device vectors ---
    cuDoubleComplex *d_v_ref = nullptr, *d_work = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_v_ref, n * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc((void**)&d_work,  n * sizeof(cuDoubleComplex)));

    // --- Host work buffers ---
    std::vector<cuDoubleComplex> h_v(n), h_y(n), h_ref(n);

    // --- RNG ---
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> G(0.0, 1.0);

    // --- Orders list ---
    std::vector<int> orders;
    for (int o = order_min; o <= order_max; o += order_step) orders.push_back(o);
    const int K = (int)orders.size();

    // --- Accumulators for mean/std (sum and sum of squares) ---
    std::vector<long double> sum_cheb(K, 0.0L), sumsq_cheb(K, 0.0L);
    std::vector<long double> sum_tay (K, 0.0L), sumsq_tay (K, 0.0L);

    // --- Trials loop: vary matrix values + input vector ---
    for (int t = 0; t < trials; ++t) {
        // random SPD values for tri-diagonal pattern
        fill_tridiag_values_random_spd(n, h_rowPtr, h_colInd, h_vals, rng);
        CHECK_CUDA(cudaMemcpy(d_vals, h_vals.data(), nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        // new random input vector
        for (int i = 0; i < n; ++i) h_v[i] = make_cuDoubleComplex(G(rng), G(rng));

        // reference: exp(+iA) v via very-high-order Chebyshev (t = -1 => exp(+iA))
        CHECK_CUDA(cudaMemcpy(d_v_ref, h_v.data(), n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        {
            int status = expiAv_chebyshev_gamma_cusparse_host(
                handle, n, nnz, order_ref,
                std::span<const int>(h_rowPtr.data(), h_rowPtr.size()),
                std::span<const int>(h_colInd.data(), h_colInd.size()),
                std::span<const cuDoubleComplex>(h_vals.data(), h_vals.size()),
                d_v_ref, -1.0
            );
            if (status != 0) {
                std::cerr << "[trial " << t << "] reference Chebyshev failed: " << status << "\n";
                cusparseDestroy(handle);
                cudaFree(d_rowPtr); cudaFree(d_colInd); cudaFree(d_vals);
                cudaFree(d_v_ref);  cudaFree(d_work);
                return 1;
            }
        }
        CHECK_CUDA(cudaMemcpy(h_ref.data(), d_v_ref, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

        // sweep Chebyshev orders
        for (int ki = 0; ki < K; ++ki) {
            int ord = orders[ki];
            CHECK_CUDA(cudaMemcpy(d_work, h_v.data(), n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            int status = expiAv_chebyshev_gamma_cusparse_host(
                handle, n, nnz, ord,
                std::span<const int>(h_rowPtr.data(), h_rowPtr.size()),
                std::span<const int>(h_colInd.data(), h_colInd.size()),
                std::span<const cuDoubleComplex>(h_vals.data(), h_vals.size()),
                d_work, -1.0
            );
            if (status != 0) continue;
            CHECK_CUDA(cudaMemcpy(h_y.data(), d_work, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            double err = rel_error_l2(h_y, h_ref);
            sum_cheb[ki]   += (long double)err;
            sumsq_cheb[ki] += (long double)err * (long double)err;
        }

        // sweep Taylor orders
        for (int ki = 0; ki < K; ++ki) {
            int ord = orders[ki];
            CHECK_CUDA(cudaMemcpy(d_work, h_v.data(), n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            int status = expiAv_taylor_cusparse(
                handle, n, nnz, ord,
                d_rowPtr, d_colInd, d_vals,
                d_work
            );
            if (status != 0) continue;
            CHECK_CUDA(cudaMemcpy(h_y.data(), d_work, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            double err = rel_error_l2(h_y, h_ref);
            sum_tay[ki]   += (long double)err;
            sumsq_tay[ki] += (long double)err * (long double)err;
        }
    }

    // --- Output CSV: method,order,mean_error,std_error ---
    std::cout << "method,order,mean_error,std_error\n";
    for (int ki = 0; ki < K; ++ki) {
        int ord = orders[ki];

        // Chebyshev
        long double mean_c = sum_cheb[ki] / (long double)trials;
        long double var_c  = std::max((long double)0.0,
                               (sumsq_cheb[ki] / (long double)trials) - mean_c * mean_c);
        long double std_c  = std::sqrt((double)var_c);
        std::cout << "chebyshev," << ord << "," << (double)mean_c << "," << (double)std_c << "\n";

        // Taylor
        long double mean_t = sum_tay[ki] / (long double)trials;
        long double var_t  = std::max((long double)0.0,
                               (sumsq_tay[ki] / (long double)trials) - mean_t * mean_t);
        long double std_t  = std::sqrt((double)var_t);
        std::cout << "taylor," << ord << "," << (double)mean_t << "," << (double)std_t << "\n";
    }

    // --- Cleanup ---
    cusparseDestroy(handle);
    cudaFree(d_rowPtr); cudaFree(d_colInd); cudaFree(d_vals);
    cudaFree(d_v_ref);  cudaFree(d_work);
    return 0;
}
