// benchmark_sparse_vs_custatevec_bulk_nostate.cpp
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <cusparse.h>
#include <custatevec.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../CudaControl/Helper.hpp"             // THROW_* / THROW_* / etc.
#include "../CuQuantumControl/ApplyGates.hpp"    // applyGatesGeneral
#include "../CuSparseControl/SparseGateBULK.hpp" // applySparseGateBulk
#include "../CuQuantumControl/Precision.hpp" // applySparseGateBulk
#include "../CuQuantumControl/StateObject.hpp"

using cplx = cuDoubleComplex;
const precision PREC = precision::bit_64;
using QState = quantumState_SV<precision::bit_64>;

static inline cplx C(double r, double i = 0.0) { return make_cuDoubleComplex(r, i); }

static inline double to_ms(std::chrono::high_resolution_clock::duration d)
{
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(d).count();
}

struct Timer
{
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    void tic() { t0 = clock::now(); }
    double toc_ms() const { return to_ms(clock::now() - t0); }
};

struct Stats
{
    double mean = 0, stdev = 0;
};
static Stats mean_stdev_ms(const std::vector<double> &v)
{
    if (v.empty())
        return {};
    double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double s2 = 0.0;
    for (double x : v)
    {
        double d = x - m;
        s2 += d * d;
    }
    s2 /= (v.size() > 1 ? (v.size() - 1) : 1);
    return {m, std::sqrt(s2)};
}

// -------------- Matrix & CSR helpers --------------

static inline cplx make_rand_cplx(std::mt19937_64 &rng, double scale = 1.0)
{
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    return C(scale * uni(rng), scale * uni(rng));
}

// Build dense d×d (row-major) matrix.
static std::vector<cplx> make_dense_matrix(int d, std::mt19937_64 &rng)
{
    std::vector<cplx> M;
    M.reserve(static_cast<size_t>(d) * d);
    for (int i = 0; i < d * d; ++i)
        M.push_back(make_rand_cplx(rng, 1.0));
    return M;
}

// Convert dense row-major matrix to CSR by keeping each entry with prob≈density.
struct CSR
{
    std::vector<int> rowPtr;  // size d+1
    std::vector<int> colInd;  // size nnz
    std::vector<cplx> values; // size nnz
    int d = 0;
};

static CSR dense_to_csr_by_density(const std::vector<cplx> &M, int d, double density, std::mt19937_64 &rng)
{
    density = std::clamp(density, 0.0, 1.0);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    CSR csr;
    csr.d = d;
    csr.rowPtr.assign(d + 1, 0);
    csr.colInd.clear();
    csr.values.clear();
    csr.colInd.reserve(size_t(density * d * d) + d);
    csr.values.reserve(csr.colInd.capacity());

    for (int r = 0; r < d; ++r)
    {
        for (int c = 0; c < d; ++c)
        {
            const cplx val = M[size_t(r) * d + c];
            if ((U(rng) < density) && (std::abs(val.x) + std::abs(val.y) > 0.0))
            {
                csr.colInd.push_back(c);
                csr.values.push_back(val);
            }
        }
        csr.rowPtr[r + 1] = static_cast<int>(csr.colInd.size());
    }
    if (csr.colInd.empty())
    {
        csr.colInd.push_back(0);
        csr.values.push_back(C(1.0, 0.0));
        csr.rowPtr[1] = 1;
        for (int r = 1; r < d; ++r)
            csr.rowPtr[r + 1] = 1;
    }
    return csr;
}

static std::vector<int> make_sequential_targets(int k)
{
    std::vector<int> t(k);
    std::iota(t.begin(), t.end(), 0);
    return t;
}

// -------------- State helpers --------------

static void fill_random_state(std::vector<cplx> &h_sv, std::mt19937_64 &rng, double scale = 1e-3)
{
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (auto &z : h_sv)
        z = C(scale * U(rng), scale * U(rng));
}

static void normalise(std::vector<cplx> &h_sv)
{
    long double accum = 0.0L;
    for (auto &z : h_sv)
    {
        long double a = z.x;
        long double b = z.y;
        accum += a * a + b * b;
    }
    if (accum == 0)
        return;
    double inv = 1.0 / std::sqrt(static_cast<double>(accum));
    for (auto &z : h_sv)
    {
        z.x *= inv;
        z.y *= inv;
    }
}

// -------------- Benchmark core --------------

struct BenchRow
{
    int nQubits;
    int kTargets;
    int d;
    double density; // [0,1]
    int nnz;
    int repeats;
    double mean_dense_ms, stdev_dense_ms;
    double mean_sparse_ms, stdev_sparse_ms;
    double speedup_sparse_vs_dense; // >1 => sparse faster
};

static BenchRow run_case(
    int nQubits,
    int kTargets,
    double density,
    int repeats,
    std::mt19937_64 &rng)
{

    const int d = 1 << kTargets;
    const size_t dim = size_t(1) << nQubits;

    // 1) Build matrices
    auto denseM = make_dense_matrix(d, rng);
    auto csr = dense_to_csr_by_density(denseM, d, density, rng);
    const int nnz = static_cast<int>(csr.colInd.size());

    // 2) Copy CSR to device
    int *d_rp = nullptr, *d_ci = nullptr;
    cplx *d_v = nullptr;
    THROW_CUDA(cudaMalloc(&d_rp, sizeof(int) * (d + 1)));
    THROW_CUDA(cudaMalloc(&d_ci, sizeof(int) * nnz));
    THROW_CUDA(cudaMalloc(&d_v, sizeof(cplx) * nnz));
    THROW_CUDA(cudaMemcpy(d_rp, csr.rowPtr.data(), sizeof(int) * (d + 1), cudaMemcpyHostToDevice));
    THROW_CUDA(cudaMemcpy(d_ci, csr.colInd.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice));
    THROW_CUDA(cudaMemcpy(d_v, csr.values.data(), sizeof(cplx) * nnz, cudaMemcpyHostToDevice));

    // 3) Allocate device state buffers
    cplx *d_sv_dense = nullptr;
    cplx *d_sv_sparse_in = nullptr;
    cplx *d_sv_sparse_out = nullptr;
    THROW_CUDA(cudaMalloc(&d_sv_dense, dim * sizeof(cplx)));
    THROW_CUDA(cudaMalloc(&d_sv_sparse_in, dim * sizeof(cplx)));
    THROW_CUDA(cudaMalloc(&d_sv_sparse_out, dim * sizeof(cplx)));

    // 4) Host initial state (randomized)
    std::vector<cplx> h_init(dim);
    fill_random_state(h_init, rng);
    normalise(h_init);

    // 5) cuSPARSE and cuStateVec handles
    cusparseHandle_t h_cusparse = nullptr;
    THROW_CUSPARSE(cusparseCreate(&h_cusparse));
    custatevecHandle_t h_custate = nullptr;
    THROW_CUSTATEVECTOR(custatevecCreate(&h_custate));

    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    // Targets & controls (no controls for perf comparability)
    const std::vector<int> targets = make_sequential_targets(kTargets);
    const std::vector<int> controls;

    // 6) Warm-up
    {
        // Dense warm-up
        THROW_CUDA(cudaMemcpy(d_sv_dense, h_init.data(), dim * sizeof(cplx), cudaMemcpyHostToDevice));
        THROW_BROAD_ERROR(applyGatesGeneral<PREC>(
            h_custate,
            nQubits,
            std::span<const cplx>(denseM.data(), denseM.size()),
            /*adjoint=*/0,
            std::span<const int>(targets.data(), targets.size()),
            std::span<const int>(controls.data(), controls.size()),
            d_sv_dense,
            extraWorkspace,
            extraWorkspaceSizeInBytes));
        THROW_CUDA(cudaDeviceSynchronize());

        // Sparse warm-up
        THROW_CUDA(cudaMemcpy(d_sv_sparse_in, h_init.data(), dim * sizeof(cplx), cudaMemcpyHostToDevice));
        THROW_CUDA(cudaMemcpy(d_sv_sparse_out, h_init.data(), dim * sizeof(cplx), cudaMemcpyHostToDevice)); // preserve untouched blocks
        THROW_BROAD_ERROR(applySparseGateBulk(
            h_cusparse,
            nQubits,
            d_rp, d_ci, d_v,
            d_sv_sparse_in,
            d_sv_sparse_out,
            targets,
            controls,
            nnz));
        THROW_CUDA(cudaDeviceSynchronize());
    }

    // 7) Timed runs
    std::vector<double> times_dense_ms, times_sparse_ms;
    times_dense_ms.reserve(repeats);
    times_sparse_ms.reserve(repeats);

    for (int r = 0; r < repeats; ++r)
    {
        // New random initial state per repeat (keeps runs honest)
        fill_random_state(h_init, rng);
        normalise(h_init);

        // Dense
        THROW_CUDA(cudaMemcpy(d_sv_dense, h_init.data(), dim * sizeof(cplx), cudaMemcpyHostToDevice));
        Timer td;
        td.tic();
        THROW_BROAD_ERROR(applyGatesGeneral<PREC>(
            h_custate,
            nQubits,
            std::span<const cplx>(denseM.data(), denseM.size()),
            /*adjoint=*/0,
            std::span<const int>(targets.data(), targets.size()),
            std::span<const int>(controls.data(), controls.size()),
            d_sv_dense,
            extraWorkspace,
            extraWorkspaceSizeInBytes));
        THROW_CUDA(cudaDeviceSynchronize());
        times_dense_ms.push_back(td.toc_ms());

        // Sparse
        THROW_CUDA(cudaMemcpy(d_sv_sparse_in, h_init.data(), dim * sizeof(cplx), cudaMemcpyHostToDevice));
        THROW_CUDA(cudaMemcpy(d_sv_sparse_out, h_init.data(), dim * sizeof(cplx), cudaMemcpyHostToDevice)); // keep other blocks
        Timer ts;
        ts.tic();
        THROW_BROAD_ERROR(applySparseGateBulk(
            h_cusparse,
            nQubits,
            d_rp, d_ci, d_v,
            d_sv_sparse_in,
            d_sv_sparse_out,
            targets,
            controls,
            nnz));
        THROW_CUDA(cudaDeviceSynchronize());
        times_sparse_ms.push_back(ts.toc_ms());
    }

    auto Sd = mean_stdev_ms(times_dense_ms);
    auto Ss = mean_stdev_ms(times_sparse_ms);

    // 8) Cleanup
    if (extraWorkspace)
    {
        cudaFree(extraWorkspace);
        extraWorkspace = nullptr;
        extraWorkspaceSizeInBytes = 0;
    }
    cusparseDestroy(h_cusparse);
    custatevecDestroy(h_custate);
    cudaFree(d_rp);
    cudaFree(d_ci);
    cudaFree(d_v);
    cudaFree(d_sv_dense);
    cudaFree(d_sv_sparse_in);
    cudaFree(d_sv_sparse_out);

    return BenchRow{
        .nQubits = nQubits,
        .kTargets = kTargets,
        .d = d,
        .density = density,
        .nnz = nnz,
        .repeats = repeats,
        .mean_dense_ms = Sd.mean,
        .stdev_dense_ms = Sd.stdev,
        .mean_sparse_ms = Ss.mean,
        .stdev_sparse_ms = Ss.stdev,
        .speedup_sparse_vs_dense = (Sd.mean > 0.0 ? (Sd.mean / Ss.mean) : 0.0)};
}

// -------------- Sweep & CSV --------------

int main_runner()
{
    const std::array<double, 12> densities = {
        1.00, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01};
    const int repeats = 7;
    const uint64_t seed = 0xC0FFEE1234ULL;

    std::ofstream csv("sparse_vs_custatevec_bulk_nostate.csv");
    if (!csv)
    {
        std::cerr << "Failed to open CSV file for writing.\n";
        return 1;
    }
    csv << "nQubits,kTargets,density,nnz,d,repeats,"
           "mean_dense_ms,stdev_dense_ms,mean_sparse_ms,stdev_sparse_ms,speedup_sparse_vs_dense\n";

    std::mt19937_64 rng(seed);

    struct Turning
    {
        int nQubits;
        int k;
        double density;
        double speedup;
    };
    std::vector<Turning> turns;

    for (int nQubits = 6; nQubits <= 20; ++nQubits)
    {
        const int kTargetsMax = std::min(nQubits, 10);
        for (int kTargets = 1; kTargets <= kTargetsMax; ++kTargets)
        {
            bool found_turn = false;

            for (double density : densities)
            {
                auto row = run_case(nQubits, kTargets, density, repeats, rng);

                // CSV
                csv << row.nQubits << ","
                    << row.kTargets << ","
                    << std::fixed << std::setprecision(2) << row.density << ","
                    << row.nnz << ","
                    << row.d << ","
                    << row.repeats << ","
                    << std::setprecision(4)
                    << row.mean_dense_ms << ","
                    << row.stdev_dense_ms << ","
                    << row.mean_sparse_ms << ","
                    << row.stdev_sparse_ms << ","
                    << std::setprecision(3) << row.speedup_sparse_vs_dense
                    << "\n";

                // Mirror to stdout
                std::cout << row.nQubits << ","
                          << row.kTargets << ","
                          << std::fixed << std::setprecision(2) << row.density << ","
                          << row.nnz << ","
                          << row.d << ","
                          << row.repeats << ","
                          << std::setprecision(4)
                          << row.mean_dense_ms << ","
                          << row.stdev_dense_ms << ","
                          << row.mean_sparse_ms << ","
                          << row.stdev_sparse_ms << ","
                          << std::setprecision(3) << row.speedup_sparse_vs_dense
                          << "\n";

                if (!found_turn && row.speedup_sparse_vs_dense > 1.0)
                {
                    turns.push_back(Turning{nQubits, kTargets, density, row.speedup_sparse_vs_dense});
                    found_turn = true;
                }
            }

            if (!found_turn)
            {
                turns.push_back(Turning{nQubits, kTargets, -1.0, 0.0});
            }
        }
    }

    std::cerr << "\n=== Turning point summary (first density where sparse > dense) ===\n";
    for (const auto &t : turns)
    {
        if (t.density < 0)
        {
            std::cerr << "nQubits=" << t.nQubits
                      << ", k=" << t.k << " (d=" << (1 << t.k)
                      << "): no sparse win in tested range.\n";
        }
        else
        {
            std::cerr << "nQubits=" << t.nQubits
                      << ", k=" << t.k << " (d=" << (1 << t.k)
                      << "): density≈" << std::fixed << std::setprecision(2) << t.density
                      << "  speedup=" << std::setprecision(3) << t.speedup << "x\n";
        }
    }

    csv.close();
    return 0;
}
