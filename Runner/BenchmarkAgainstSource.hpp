// bench_simple_user_vs_cusv.cpp
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

using QState = quantumState_SV<precision::bit_64>;
using cplx = cuDoubleComplex;

// ===================== Toggle Flags (override with -D on compile) =====================
#ifndef BENCH_ENABLE_YOURS
#define BENCH_ENABLE_YOURS 1
#endif

#ifndef BENCH_ENABLE_CUSTATEVEC_DEVICE
#define BENCH_ENABLE_CUSTATEVEC_DEVICE 1
#endif

#ifndef BENCH_ENABLE_CUSTATEVEC_UNIFIED
#define BENCH_ENABLE_CUSTATEVEC_UNIFIED 1
#endif

#ifndef BENCH_ENABLE_FUNCTIONAL_DEVICE
#define BENCH_ENABLE_FUNCTIONAL_DEVICE 1
#endif

#ifndef BENCH_ENABLE_FUNCTIONAL_UNIFIED
#define BENCH_ENABLE_FUNCTIONAL_UNIFIED 1
#endif

// Whether to run a short warmup loop before timing
#ifndef BENCH_ENABLE_WARMUP
#define BENCH_ENABLE_WARMUP 1
#endif

// Whether to cudaDeviceSynchronize() after timed loops (keeps timing honest)
#ifndef BENCH_ENABLE_SYNC_AFTER
#define BENCH_ENABLE_SYNC_AFTER 1
#endif

// Print CSV header once at start
#ifndef BENCH_PRINT_CSV_HEADER
#define BENCH_PRINT_CSV_HEADER 1
#endif

// Default number of passes (can override with -DPASSES_DEFAULT=...)
#ifndef PASSES_DEFAULT
#define PASSES_DEFAULT 10
#endif
// ================================================================================

static inline cplx C(double r, double i = 0) { return make_cuDoubleComplex(r, i); }
static inline double ms(auto d)
{
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(d).count();
}

int runBench(int argc, char **argv)
{
    int n_min = 16, n_max = 28, step = 2, repeats = 3;
    if (argc >= 5)
    {
        n_min = std::atoi(argv[1]);
        n_max = std::atoi(argv[2]);
        step = std::atoi(argv[3]);
        repeats = std::atoi(argv[4]);
    }

    // gate defs (row-major)
    const cplx X1[4] = {C(0), C(1), C(1), C(0)};
    constexpr int passes = PASSES_DEFAULT;

#if BENCH_PRINT_CSV_HEADER
    std::cout << "backend,n,passes,repeats,mean_ms,stdev_ms\n";
#endif

    for (int n = n_min; n <= n_max; n += step)
    {
#if BENCH_ENABLE_YOURS
        // ------------ YOUR PACKAGE ------------
        {
            QState qs(n);
            qs.zero_state();
            qs.prefetchToDevice();

#if BENCH_ENABLE_WARMUP
            // warm-up (not timed)
            for (int rep = 0; rep < 1; ++rep)
            {
                // X on all
                for (int q = 0; q < n; ++q)
                {
                    qs.X({q});
                }
                // CX on (q,q+1)
                for (int q = 0; q < n - 1; ++q)
                {
                    qs.X({q}, {q + 1});
                }
            }
            cudaDeviceSynchronize();
#endif

            std::vector<double> trials;
            for (int r = 0; r < repeats; ++r)
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int rep = 0; rep < passes; ++rep)
                {
                    for (int q = 0; q < n; ++q)
                    {
                        qs.X({q});
                    }
                    for (int q = 0; q < n - 1; ++q)
                    {
                        qs.X({q}, {q + 1});
                    }
                }
#if BENCH_ENABLE_SYNC_AFTER
                cudaDeviceSynchronize();
#endif
                trials.push_back(ms(std::chrono::high_resolution_clock::now() - t0));
            }
            double mu = std::accumulate(trials.begin(), trials.end(), 0.0) / trials.size();
            double s2 = 0;
            for (double x : trials)
            {
                double d = x - mu;
                s2 += d * d;
            }
            double sd = std::sqrt(s2 / std::max(1, (int)trials.size() - 1));
            std::cout << "yours," << n << "," << passes << "," << repeats << ","
                      << std::fixed << std::setprecision(3) << mu << "," << sd << "\n";
        }
#endif // BENCH_ENABLE_YOURS

#if BENCH_ENABLE_CUSTATEVEC_DEVICE
        // ------------ cuStateVec (device) ------------
        {
            custatevecHandle_t h = nullptr;
            custatevecCreate(&h);

            const size_t dim = size_t(1) << n;
            cplx *d_sv = nullptr;
            cudaMalloc(&d_sv, dim * sizeof(cplx));
            cudaMemset(d_sv, 0, dim * sizeof(cplx));
            cplx one = C(1, 0);
            cudaMemcpy(d_sv, &one, sizeof(cplx), cudaMemcpyHostToDevice);

            // --- Workspace sizing ---
            size_t ws_x = 0, ws_cx = 0, ws = 0;
            // 1-qubit X gate workspace
            {
                const int tgt[1] = {0};
                custatevecApplyMatrixGetWorkspaceSize(
                    h, CUDA_C_64F, (int)n,
                    X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                    0, 1, 0, CUSTATEVEC_COMPUTE_64F, &ws_x);
            }
            // Controlled-X workspace
            {
                const int tgt[1] = {0};
                const int ctl[1] = {1};
                const int ctlv[1] = {1};
                custatevecApplyMatrixGetWorkspaceSize(
                    h, CUDA_C_64F, (int)n,
                    X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                    0, 1, 1, CUSTATEVEC_COMPUTE_64F, &ws_cx);
            }
            ws = std::max(ws_x, ws_cx);
            void *d_work = nullptr;
            if (ws > 0)
                cudaMalloc(&d_work, ws);

#if BENCH_ENABLE_WARMUP
            // --- Warm-up ---
            for (int rep = 0; rep < 1; ++rep)
            {
                for (int q = 0; q < n; ++q)
                {
                    const int tgt[1] = {q};
                    custatevecApplyMatrix(h, d_sv, CUDA_C_64F, n,
                                          X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                          tgt, 1, nullptr, nullptr, 0,
                                          CUSTATEVEC_COMPUTE_64F, d_work, ws);
                }
                for (int q = 0; q < n - 1; ++q)
                {
                    const int tgt[1] = {q};
                    const int ctl[1] = {q + 1};
                    const int ctlv[1] = {1};
                    custatevecApplyMatrix(h, d_sv, CUDA_C_64F, n,
                                          X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                          tgt, 1, ctl, ctlv, 1,
                                          CUSTATEVEC_COMPUTE_64F, d_work, ws);
                }
            }
            cudaDeviceSynchronize();
#endif

            // --- Timed runs ---
            std::vector<double> trials;
            for (int r = 0; r < repeats; ++r)
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int rep = 0; rep < passes; ++rep)
                {
                    for (int q = 0; q < n; ++q)
                    {
                        const int tgt[1] = {q};
                        custatevecApplyMatrix(h, d_sv, CUDA_C_64F, n,
                                              X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                              tgt, 1, nullptr, nullptr, 0,
                                              CUSTATEVEC_COMPUTE_64F, d_work, ws);
                    }
                    for (int q = 0; q < n - 1; ++q)
                    {
                        const int tgt[1] = {q};
                        const int ctl[1] = {q + 1};
                        const int ctlv[1] = {1};
                        custatevecApplyMatrix(h, d_sv, CUDA_C_64F, n,
                                              X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                              tgt, 1, ctl, ctlv, 1,
                                              CUSTATEVEC_COMPUTE_64F, d_work, ws);
                    }
                }
#if BENCH_ENABLE_SYNC_AFTER
                cudaDeviceSynchronize();
#endif
                trials.push_back(ms(std::chrono::high_resolution_clock::now() - t0));
            }

            double mu = std::accumulate(trials.begin(), trials.end(), 0.0) / trials.size();
            double s2 = 0;
            for (double x : trials)
            {
                double d = x - mu;
                s2 += d * d;
            }
            double sd = std::sqrt(s2 / std::max(1, (int)trials.size() - 1));
            std::cout << "custatevecDevice," << n << "," << passes << "," << repeats
                      << "," << std::fixed << std::setprecision(3) << mu << "," << sd << "\n";

            if (d_work)
                cudaFree(d_work);
            cudaFree(d_sv);
            custatevecDestroy(h);
        }
#endif // BENCH_ENABLE_CUSTATEVEC_DEVICE

#if BENCH_ENABLE_CUSTATEVEC_UNIFIED
        // ------------ cuStateVec (unified) ------------
        {
            custatevecHandle_t h = nullptr;
            custatevecCreate(&h);

            const size_t dim = size_t(1) << n;
            cplx *d_sv = nullptr;
            cudaMallocManaged(&d_sv, dim * sizeof(cplx));
            cudaMemset(d_sv, 0, dim * sizeof(cplx));
            cplx one = C(1, 0);
            cudaMemcpy(d_sv, &one, sizeof(cplx), cudaMemcpyHostToDevice);

            // --- Workspace sizing ---
            size_t ws_x = 0, ws_cx = 0, ws = 0;
            {
                const int tgt[1] = {0};
                custatevecApplyMatrixGetWorkspaceSize(
                    h, CUDA_C_64F, (int)n,
                    X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                    0, 1, 0, CUSTATEVEC_COMPUTE_64F, &ws_x);
            }
            {
                const int tgt[1] = {0};
                const int ctl[1] = {1};
                const int ctlv[1] = {1};
                custatevecApplyMatrixGetWorkspaceSize(
                    h, CUDA_C_64F, (int)n,
                    X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                    0, 1, 1, CUSTATEVEC_COMPUTE_64F, &ws_cx);
            }
            ws = std::max(ws_x, ws_cx);
            void *d_work = (ws > 0 ? nullptr : nullptr);
            if (ws > 0)
                cudaMalloc(&d_work, ws);

#if BENCH_ENABLE_WARMUP
            for (int rep = 0; rep < 1; ++rep)
            {
                for (int q = 0; q < n; ++q)
                {
                    const int tgt[1] = {q};
                    custatevecApplyMatrix(h, d_sv, CUDA_C_64F, n,
                                          X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                          tgt, 1, nullptr, nullptr, 0,
                                          CUSTATEVEC_COMPUTE_64F, d_work, ws);
                }
                for (int q = 0; q < n - 1; ++q)
                {
                    const int tgt[1] = {q};
                    const int ctl[1] = {q + 1};
                    const int ctlv[1] = {1};
                    custatevecApplyMatrix(h, d_sv, CUDA_C_64F, n,
                                          X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                          tgt, 1, ctl, ctlv, 1,
                                          CUSTATEVEC_COMPUTE_64F, d_work, ws);
                }
            }
            cudaDeviceSynchronize();
#endif

            std::vector<double> trials;
            for (int r = 0; r < repeats; ++r)
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int rep = 0; rep < passes; ++rep)
                {
                    for (int q = 0; q < n; ++q)
                    {
                        const int tgt[1] = {q};
                        custatevecApplyMatrix(h, d_sv, CUDA_C_64F, n,
                                              X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                              tgt, 1, nullptr, nullptr, 0,
                                              CUSTATEVEC_COMPUTE_64F, d_work, ws);
                    }
                    for (int q = 0; q < n - 1; ++q)
                    {
                        const int tgt[1] = {q};
                        const int ctl[1] = {q + 1};
                        const int ctlv[1] = {1};
                        custatevecApplyMatrix(h, d_sv, CUDA_C_64F, n,
                                              X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                              tgt, 1, ctl, ctlv, 1,
                                              CUSTATEVEC_COMPUTE_64F, d_work, ws);
                    }
                }
#if BENCH_ENABLE_SYNC_AFTER
                cudaDeviceSynchronize();
#endif
                trials.push_back(ms(std::chrono::high_resolution_clock::now() - t0));
            }
            double mu = std::accumulate(trials.begin(), trials.end(), 0.0) / trials.size();
            double s2 = 0;
            for (double x : trials)
            {
                double d = x - mu;
                s2 += d * d;
            }
            double sd = std::sqrt(s2 / std::max(1, (int)trials.size() - 1));
            std::cout << "custatevecUnified," << n << "," << passes << "," << repeats
                      << "," << std::fixed << std::setprecision(3) << mu << "," << sd << "\n";

            if (d_work)
                cudaFree(d_work);
            cudaFree(d_sv);
            custatevecDestroy(h);
        }
#endif // BENCH_ENABLE_CUSTATEVEC_UNIFIED

#if BENCH_ENABLE_FUNCTIONAL_DEVICE
        // ------------ functional (device) ------------
        {
            custatevecHandle_t h = nullptr;
            custatevecCreate(&h);
            const size_t dim = size_t(1) << n;
            cplx *d_sv = nullptr;
            cudaMalloc(&d_sv, dim * sizeof(cplx));
            cudaMemset(d_sv, 0, dim * sizeof(cplx));
            cplx one = C(1, 0);
            cudaMemcpy(d_sv, &one, sizeof(cplx), cudaMemcpyHostToDevice);
            void *extraWorkspace = nullptr;
            size_t extraWorkspaceSizeInBytes = 0;

#if BENCH_ENABLE_WARMUP
            for (int rep = 0; rep < 1; ++rep)
            {
                for (int q = 0; q < n; ++q)
                {
                    applyX(h, n, 0, std::array<int, 1>{q}, d_sv, extraWorkspace, extraWorkspaceSizeInBytes);
                }
                for (int q = 0; q < n - 1; ++q)
                {
                    applyX(h, n, 0, std::array<int, 1>{q}, std::array<int, 1>{q + 1}, d_sv, extraWorkspace, extraWorkspaceSizeInBytes);
                }
            }
            cudaDeviceSynchronize();
#endif

            std::vector<double> trials;
            for (int r = 0; r < repeats; ++r)
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int rep = 0; rep < passes; ++rep)
                {
                    for (int q = 0; q < n; ++q)
                    {
                        applyX(h, n, 0, std::array<int, 1>{q}, d_sv, extraWorkspace, extraWorkspaceSizeInBytes);
                    }
                    for (int q = 0; q < n - 1; ++q)
                    {
                        applyX(h, n, 0, std::array<int, 1>{q}, std::array<int, 1>{q + 1}, d_sv, extraWorkspace, extraWorkspaceSizeInBytes);
                    }
                }
#if BENCH_ENABLE_SYNC_AFTER
                cudaDeviceSynchronize();
#endif
                trials.push_back(ms(std::chrono::high_resolution_clock::now() - t0));
            }
            double mu = std::accumulate(trials.begin(), trials.end(), 0.0) / trials.size();
            double s2 = 0;
            for (double x : trials)
            {
                double d = x - mu;
                s2 += d * d;
            }
            double sd = std::sqrt(s2 / std::max(1, (int)trials.size() - 1));
            std::cout << "functionalDevice," << n << "," << passes << "," << repeats
                      << "," << std::fixed << std::setprecision(3) << mu << "," << sd << "\n";

            cudaFree(d_sv);
            custatevecDestroy(h);
            if (extraWorkspace != nullptr)
            {
                THROW_CUDA(cudaFree(extraWorkspace));
                extraWorkspaceSizeInBytes = 0;
                extraWorkspace = nullptr;
            }
        }
#endif // BENCH_ENABLE_FUNCTIONAL_DEVICE

#if BENCH_ENABLE_FUNCTIONAL_UNIFIED
        // ------------ functional (unified) ------------
        {
            custatevecHandle_t h = nullptr;
            custatevecCreate(&h);
            const size_t dim = size_t(1) << n;
            cplx *d_sv = nullptr;
            cudaMallocManaged(&d_sv, dim * sizeof(cplx));
            cudaMemset(d_sv, 0, dim * sizeof(cplx));
            cplx one = C(1, 0);
            cudaMemcpy(d_sv, &one, sizeof(cplx), cudaMemcpyHostToDevice);
            void *extraWorkspace = nullptr;
            size_t extraWorkspaceSizeInBytes = 0;

#if BENCH_ENABLE_WARMUP
            for (int rep = 0; rep < 1; ++rep)
            {
                for (int q = 0; q < n; ++q)
                {
                    applyX(h, n, 0, std::array<int, 1>{q}, d_sv, extraWorkspace, extraWorkspaceSizeInBytes);
                }
                for (int q = 0; q < n - 1; ++q)
                {
                    applyX(h, n, 0, std::array<int, 1>{q}, std::array<int, 1>{q + 1}, d_sv, extraWorkspace, extraWorkspaceSizeInBytes);
                }
            }
            cudaDeviceSynchronize();
#endif

            std::vector<double> trials;
            for (int r = 0; r < repeats; ++r)
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int rep = 0; rep < passes; ++rep)
                {
                    for (int q = 0; q < n; ++q)
                    {
                        applyX(h, n, 0, std::array<int, 1>{q}, d_sv, extraWorkspace, extraWorkspaceSizeInBytes);
                    }
                    for (int q = 0; q < n - 1; ++q)
                    {
                        applyX(h, n, 0, std::array<int, 1>{q}, std::array<int, 1>{q + 1}, d_sv, extraWorkspace, extraWorkspaceSizeInBytes);
                    }
                }
#if BENCH_ENABLE_SYNC_AFTER
                cudaDeviceSynchronize();
#endif
                trials.push_back(ms(std::chrono::high_resolution_clock::now() - t0));
            }
            double mu = std::accumulate(trials.begin(), trials.end(), 0.0) / trials.size();
            double s2 = 0;
            for (double x : trials)
            {
                double d = x - mu;
                s2 += d * d;
            }
            double sd = std::sqrt(s2 / std::max(1, (int)trials.size() - 1));
            std::cout << "functionalUnified," << n << "," << passes << "," << repeats
                      << "," << std::fixed << std::setprecision(3) << mu << "," << sd << "\n";

            cudaFree(d_sv);
            custatevecDestroy(h);
            if (extraWorkspace != nullptr)
            {
                THROW_CUDA(cudaFree(extraWorkspace));
                extraWorkspaceSizeInBytes = 0;
                extraWorkspace = nullptr;
            }
        }
#endif // BENCH_ENABLE_FUNCTIONAL_UNIFIED

    } // for n

    return 0;
}
