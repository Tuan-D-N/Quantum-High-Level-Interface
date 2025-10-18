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
int runCheby()
{
    // --- Settings ---
    const int n            =256; // matrix size
    const int order_min    =2;
    const int order_max    =64;
    const int order_step   =2;
    const int order_ref    =512; // high-order Chebyshev reference
    const unsigned seed    = 42u;

    // --- cuSPARSE handle ---
    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // --- Build CSR on host ---
    std::vector<int> h_rowPtr, h_colInd;
    std::vector<cuDoubleComplex> h_vals;
    build_tridiag_laplacian(n, h_rowPtr, h_colInd, h_vals);
    const int nnz = (int)h_colInd.size();

    // --- Copy CSR to device (for Taylor) ---
    int *d_rowPtr=nullptr, *d_colInd=nullptr;
    cuDoubleComplex *d_vals=nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_rowPtr, (n+1)*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_colInd, nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals,   nnz*sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMemcpy(d_rowPtr, h_rowPtr.data(), (n+1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, h_colInd.data(), nnz*sizeof(int),   cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals,   h_vals.data(),   nnz*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // --- Random input vector v on host ---
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0,1.0);
    std::vector<cuDoubleComplex> h_v(n);
    for(int i=0;i<n;++i){ h_v[i] = make_cuDoubleComplex(g(rng), g(rng)); }

    // --- Device buffers ---
    cuDoubleComplex *d_v_ref=nullptr, *d_work=nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_v_ref, n*sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc((void**)&d_work,  n*sizeof(cuDoubleComplex)));

    // --- Compute reference: exp(+iA) v via very-high-order Chebyshev ---
    CHECK_CUDA(cudaMemcpy(d_v_ref, h_v.data(), n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    {
        // t = -1 for exp(+iA) according to your function’s comment
        int status = expiAv_chebyshev_gamma_cusparse_host(
            handle, n, nnz, order_ref,
            std::span<const int>(h_rowPtr.data(), h_rowPtr.size()),
            std::span<const int>(h_colInd.data(), h_colInd.size()),
            std::span<const cuDoubleComplex>(h_vals.data(), h_vals.size()),
            d_v_ref, /* in/out */
            -1.0
        );
        if (status!=0) {
            std::cerr<<"Reference Chebyshev failed with status "<<status<<"\n";
            return 1;
        }
    }
    // Pull reference back
    std::vector<cuDoubleComplex> h_ref(n);
    CHECK_CUDA(cudaMemcpy(h_ref.data(), d_v_ref, n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    // --- Print CSV header ---
    std::cout << "method,order,error\n";

    // --- Sweep orders for Chebyshev ---
    for(int ord=order_min; ord<=order_max; ord+=order_step){
        // reset input
        CHECK_CUDA(cudaMemcpy(d_work, h_v.data(), n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        int status = expiAv_chebyshev_gamma_cusparse_host(
            handle, n, nnz, ord,
            std::span<const int>(h_rowPtr.data(), h_rowPtr.size()),
            std::span<const int>(h_colInd.data(), h_colInd.size()),
            std::span<const cuDoubleComplex>(h_vals.data(), h_vals.size()),
            d_work, /* in/out */
            -1.0    /* exp(+iA) */
        );
        if (status!=0) {
            std::cerr<<"Chebyshev(order="<<ord<<") failed with status "<<status<<"\n";
            std::cout<<"chebyshev,"<<ord<<",nan\n";
            continue;
        }
        std::vector<cuDoubleComplex> h_y(n);
        CHECK_CUDA(cudaMemcpy(h_y.data(), d_work, n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        double err = rel_error_l2(h_y, h_ref);
        std::cout<<"chebyshev,"<<ord<<","<<err<<"\n";
    }

    // --- Sweep orders for Taylor ---
    for(int ord=order_min; ord<=order_max; ord+=order_step){
        // reset input
        CHECK_CUDA(cudaMemcpy(d_work, h_v.data(), n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        int status = expiAv_taylor_cusparse(
            handle, n, nnz, ord,
            d_rowPtr, d_colInd, d_vals,
            d_work /* assuming in/out */
        );
        if (status!=0) {
            std::cerr<<"Taylor(order="<<ord<<") failed with status "<<status<<"\n";
            std::cout<<"taylor,"<<ord<<",nan\n";
            continue;
        }
        std::vector<cuDoubleComplex> h_y(n);
        CHECK_CUDA(cudaMemcpy(h_y.data(), d_work, n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        double err = rel_error_l2(h_y, h_ref);
        std::cout<<"taylor,"<<ord<<","<<err<<"\n";
    }

    // --- Cleanup ---
    cusparseDestroy(handle);
    cudaFree(d_rowPtr); cudaFree(d_colInd); cudaFree(d_vals);
    cudaFree(d_v_ref);  cudaFree(d_work);
    return 0;
}