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
    // const cplx CNOT2[16] = {
    //     C(1),C(0),C(0),C(0),
    //     C(0),C(1),C(0),C(0),
    //     C(0),C(0),C(0),C(1),
    //     C(0),C(0),C(1),C(0)
    // };
    constexpr int passes = 10; // do X pass + CX pass, repeated 10 times

    std::cout << "backend,n,passes,repeats,mean_ms,stdev_ms\n";

    for (int n = n_min; n <= n_max; n += step)
    {
        // ------------ YOUR PACKAGE ------------
        {
            QState qs(n);
            qs.zero_state();
            qs.prefetchToDevice();

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
                cudaDeviceSynchronize();
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
            std::cout << "yours," << n << "," << passes << "," << repeats << "," << std::fixed << std::setprecision(3) << mu << "," << sd << "\n";
        }

        // ------------ cuStateVec ------------
        {
            custatevecHandle_t h = nullptr;
            custatevecCreate(&h);
            const size_t dim = size_t(1) << n;
            cplx *d_sv = nullptr;
            cudaMalloc(&d_sv, dim * sizeof(cplx));
            cudaMemset(d_sv, 0, dim * sizeof(cplx));
            cplx one = C(1, 0);
            cudaMemcpy(d_sv, &one, sizeof(cplx), cudaMemcpyHostToDevice);

            // warm-up
            for (int rep = 0; rep < 1; ++rep)
            {
                for (int q = 0; q < n; ++q)
                {
                    int tgt[1] = {q};
                    custatevecApplyMatrix(h, d_sv, CUDA_C_64F, n,
                                          X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                          tgt, 1, nullptr, nullptr, 0,
                                          CUSTATEVEC_COMPUTE_64F, nullptr, 0);
                }
                for (int q = 0; q < n - 1; ++q)
                {
                    int tgt[2] = {q};
                    int ctl[1] = {q + 1};
                    int ctlv[1] = {1};
                    custatevecApplyMatrix(h, d_sv, CUDA_C_64F, n,
                                          X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                          tgt, 1, ctl, ctlv, 2,
                                          CUSTATEVEC_COMPUTE_64F, nullptr, 0);
                }
            }
            cudaDeviceSynchronize();

            std::vector<double> trials;
            for (int r = 0; r < repeats; ++r)
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int rep = 0; rep < passes; ++rep)
                {
                    for (int q = 0; q < n; ++q)
                    {
                        int tgt[1] = {q};
                        custatevecApplyMatrix(h, d_sv, CUDA_C_64F, n,
                                              X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                              tgt, 1, nullptr, nullptr, 0,
                                              CUSTATEVEC_COMPUTE_64F, nullptr, 0);
                    }
                    for (int q = 0; q < n - 1; ++q)
                    {
                        int tgt[2] = {q};
                        int ctl[1] = {q + 1};
                        int ctlv[1] = {1};
                        custatevecApplyMatrix(h, d_sv, CUDA_C_64F, n,
                                              X1, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                                              tgt, 1, ctl, ctlv, 2,
                                              CUSTATEVEC_COMPUTE_64F, nullptr, 0);
                    }
                }
                cudaDeviceSynchronize();
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
            std::cout << "custatevec," << n << "," << passes << "," << repeats << "," << std::fixed << std::setprecision(3) << mu << "," << sd << "\n";

            cudaFree(d_sv);
            custatevecDestroy(h);
        }
        // ------------ functional ------------
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
            // warm-up
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
                cudaDeviceSynchronize();
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
            std::cout << "functional," << n << "," << passes << "," << repeats << "," << std::fixed << std::setprecision(3) << mu << "," << sd << "\n";

            cudaFree(d_sv);
            custatevecDestroy(h);
            if (extraWorkspace != nullptr)
            {
                THROW_CUDA(cudaFree(extraWorkspace));
                extraWorkspaceSizeInBytes = 0;
                extraWorkspace = nullptr;
            }
        }
    }
    return 0;
}
