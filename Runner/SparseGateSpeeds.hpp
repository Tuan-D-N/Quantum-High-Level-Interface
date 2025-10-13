// benchmark_sparse_vs_dense.cpp
#include <cuda_runtime_api.h>
#include <cuComplex.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "../CuQuantumControl/StateObject.hpp"
#include "../functionality/SquareNorm.hpp"
#include "../CuSparseControl/SparseDenseConvert.hpp"

// If your precision enum/name differs, change here:
using QState = quantumState_SV<precision::bit_64>;
using cplx   = cuDoubleComplex;

// ===================== UTILITIES =====================

static inline cplx make_rand_cplx(std::mt19937_64 &rng, double scale=1.0) {
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    return make_cuDoubleComplex(scale*uni(rng), scale*uni(rng));
}

static inline double to_ms(std::chrono::high_resolution_clock::duration d) {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(d).count();
}

struct Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    void tic() { t0 = clock::now(); }
    double toc_ms() const { return to_ms(clock::now() - t0); }
};

struct Stats { double mean=0, stdev=0; };
static Stats mean_stdev_ms(const std::vector<double>& v) {
    if (v.empty()) return {};
    double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double s2 = 0.0;
    for (double x: v) { double d = x - m; s2 += d*d; }
    s2 /= (v.size() > 1 ? (v.size()-1) : 1);
    return {m, std::sqrt(s2)};
}

// Build a dense d×d matrix (row-major) for the target gate.
// Values are random; we don’t enforce unitarity since we only benchmark.
static std::vector<cplx> make_dense_matrix(int d, std::mt19937_64 &rng) {
    std::vector<cplx> M;
    M.reserve(static_cast<size_t>(d)*d);
    for (int i = 0; i < d*d; ++i) M.push_back(make_rand_cplx(rng, 1.0));
    return M;
}

// Build a *row-sorted* CSR with approximate density in (0,1].
// Ensures no duplicate column indices per row.
struct CSR {
    std::vector<int> rowPtr;   // size d+1
    std::vector<int> colInd;   // size nnz
    std::vector<cplx> values;  // size nnz
    int d = 0;
};
static CSR make_random_csr(int d, double density, std::mt19937_64 &rng) {
    density = std::clamp(density, 0.0, 1.0);
    const int nnz_target = std::max(1, static_cast<int>(std::llround(density * d * d)));
    std::vector<int> rowPtr(d+1, 0), colInd;
    std::vector<cplx> values;
    colInd.reserve(nnz_target);
    values.reserve(nnz_target);

    std::uniform_int_distribution<int> col_pick(0, d-1);

    int nnz_so_far = 0;
    for (int r = 0; r < d; ++r) {
        // approx even share per row; leave leftovers for later rows
        int rows_left = d - r;
        int want_this_row = std::max(0, (nnz_target - nnz_so_far) / rows_left);
        // to avoid zero-row issue when density small:
        if (want_this_row==0 && nnz_so_far < nnz_target) want_this_row = 1;

        std::unordered_set<int> used;
        used.reserve(static_cast<size_t>(want_this_row*1.3)+1);
        for (int k = 0; k < want_this_row && nnz_so_far < nnz_target; ++k) {
            int c;
            do { c = col_pick(rng); } while (used.count(c));
            used.insert(c);
            colInd.push_back(c);
            values.push_back(make_rand_cplx(rng, 1.0));
            ++nnz_so_far;
        }
        // sort row’s columns
        const int row_begin = colInd.size() - static_cast<int>(used.size());
        std::sort(colInd.begin() + row_begin, colInd.end());
        rowPtr[r+1] = static_cast<int>(colInd.size());
    }
    CSR csr;
    csr.rowPtr = std::move(rowPtr);
    csr.colInd = std::move(colInd);
    csr.values = std::move(values);
    csr.d = d;
    return csr;
}

// Make a “bit-ordering” list [0,1,2,...,k-1] for target qubits for convenience.
// The gate acts on these target qubits.
static std::vector<int> make_sequential_targets(int k) {
    std::vector<int> t(k);
    std::iota(t.begin(), t.end(), 0);
    return t;
}

// Fill the state |0...0> (we’ll also randomize amplitudes a bit to avoid cache effects).
static void randomize_state(QState &qs, std::mt19937_64& rng, double scale=1e-3) {
    auto sv = qs.getStateVector();
    for (auto &z : sv) {
        z = make_cuDoubleComplex(scale*std::uniform_real_distribution<double>(-1,1)(rng),
                                 scale*std::uniform_real_distribution<double>(-1,1)(rng));
    }
    qs.normalise_SV();
}

// ===================== BENCH CORE =====================

struct BenchRow {
    int nQubits;
    int kTargets;
    int d;
    double density;     // in [0,1]
    int nnz;
    int repeats;
    double mean_dense_ms, stdev_dense_ms;
    double mean_sparse_ms, stdev_sparse_ms;
    double speedup_sparse_vs_dense; // >1 means sparse is faster
};

static BenchRow run_case(QState& qs,
                         int nQubits,
                         int kTargets,
                         double density,
                         int repeats,
                         std::mt19937_64& rng)
{
    const int d = 1 << kTargets;
    // Targets: first k qubits for simplicity (adjust if you prefer different indices):
    std::vector<int> targets = make_sequential_targets(kTargets);

    // ----- Build gates -----
    auto denseM = make_dense_matrix(d, rng);
    auto csr = make_random_csr(d, density, rng);
    const int nnz = static_cast<int>(csr.colInd.size());

    // ----- Warm-up both paths once -----
    {
        // Dense
        qs.applyArbitaryGate(std::span<const int>(targets.data(), targets.size()),
                             std::span<const int>(),  // controls empty
                             std::span<const cplx>(denseM.data(), denseM.size()));
        // Sparse
        qs.applySparseMatrix(std::span<int>(csr.rowPtr.data(), csr.rowPtr.size()),
                             std::span<int>(csr.colInd.data(), csr.colInd.size()),
                             std::span<cplx>(csr.values.data(), csr.values.size()));
    }

    // ----- Timing -----
    std::vector<double> times_dense_ms, times_sparse_ms;
    times_dense_ms.reserve(repeats);
    times_sparse_ms.reserve(repeats);

    for (int r = 0; r < repeats; ++r) {
        // Reset state a little bit between runs to avoid “too stable” cache reuse
        randomize_state(qs, rng);

        // Dense
        Timer td; td.tic();
        qs.applyArbitaryGate(std::span<const int>(targets.data(), targets.size()),
                             std::span<const int>(),
                             std::span<const cplx>(denseM.data(), denseM.size()));
        times_dense_ms.push_back(td.toc_ms());

        // Sparse
        Timer ts; ts.tic();
        qs.applySparseMatrix(std::span<int>(csr.rowPtr.data(), csr.rowPtr.size()),
                             std::span<int>(csr.colInd.data(), csr.colInd.size()),
                             std::span<cplx>(csr.values.data(), csr.values.size()));
        times_sparse_ms.push_back(ts.toc_ms());
    }

    auto Sd = mean_stdev_ms(times_dense_ms);
    auto Ss = mean_stdev_ms(times_sparse_ms);

    return BenchRow {
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
        .speedup_sparse_vs_dense = (Sd.mean > 0.0 ? (Sd.mean / Ss.mean) : 0.0)
    };
}

// ===================== MAIN SWEEP =====================

int main_runner() {
    // ------- Config -------
    const int nQubits_system = 20;        // size of the *full* statevector (adjust to taste)
    const std::array<int,6> k_list = {1, 2, 3, 4, 5, 6};  // gate sizes -> d=2^k
    const std::array<double,12> densities = {
        1.00, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01
    };
    const int repeats = 7;                 // per (k, density)
    const uint64_t seed = 0xC0FFEE1234ULL; // deterministic

    // ------- State init -------
    QState qs(nQubits_system);
    qs.zero_state();
    qs.prefetchToDevice();  // for fairness: ensure resident
    std::mt19937_64 rng(seed);
    randomize_state(qs, rng);

    // CSV header
    std::cout << "nQubits,kTargets,density,nnz,d,repeats,"
                 "mean_dense_ms,stdev_dense_ms,mean_sparse_ms,stdev_sparse_ms,speedup_sparse_vs_dense\n";

    // Track turning points per k: first density where sparse beats dense (speedup>1)
    struct Turning { int k; double density; double speedup; };
    std::vector<Turning> turns;

    for (int kTargets : k_list) {
        bool found_turn=false;
        for (double density : densities) {
            auto row = run_case(qs, nQubits_system, kTargets, density, repeats, rng);
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

            if (!found_turn && row.speedup_sparse_vs_dense > 1.0) {
                turns.push_back(Turning{ kTargets, density, row.speedup_sparse_vs_dense });
                found_turn = true;
            }
        }
        if (!found_turn) {
            turns.push_back(Turning{ kTargets, -1.0, 0.0 }); // no win found
        }
    }

    // Summary
    std::cerr << "\n=== Turning point summary (first density where sparse > dense) ===\n";
    for (const auto& t : turns) {
        if (t.density < 0) {
            std::cerr << "k=" << t.k << " (d=" << (1<<t.k) << "): no sparse win in tested range.\n";
        } else {
            std::cerr << "k=" << t.k << " (d=" << (1<<t.k)
                      << "): density≈" << std::fixed << std::setprecision(2) << t.density
                      << "  speedup=" << std::setprecision(3) << t.speedup << "x\n";
        }
    }

    qs.prefetchToCPU(); // optional
    return 0;
}
