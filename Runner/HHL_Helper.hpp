#pragma once
#include <vector>
#include <span>
#include <cmath>
#include <optional>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <cassert>
#include <iomanip>
#include "../CuQuantumControl/StateObject.hpp"

// Your simulator
template <precision P>
class quantumState_SV;

// ----------------------------- Utilities -----------------------------

struct HHLParams
{
    int pe_qubits = 6;          // phase-estimation register size p
    double t0 = 1.0;            // base evolution time
    int order = 30;             // fixed polynomial order for both methods
    bool use_chebyshev = true;  // else Taylor
    double compare_eps = 1e-10; // optional for sanity checks
};

struct HHLBenchResult
{
    float ms_phase_estimate = 0.0f; // time spent in controlled e^{+iA t} blocks
    float ms_inv_qft = 0.0f;
    float ms_total = 0.0f;
    double post_select_prob = 0.0; // filled if you do the reciprocal+project
};

// Little helper for cudaEvent timing
struct CudaTimer
{
    cudaEvent_t start{}, stop{};
    CudaTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~CudaTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void tic() { cudaEventRecord(start); }
    float toc()
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// Map a contiguous block of “phase” qubits: [phase_base, phase_base+pe_qubits)
inline std::vector<int> phase_range(int phase_base, int pe_qubits)
{
    std::vector<int> q(pe_qubits);
    for (int i = 0; i < pe_qubits; ++i)
        q[i] = phase_base + i;
    return q;
}

// Inverse QFT on bit-order [q0(LSB), q1, ..., q_{p-1}(MSB)]
template <precision P>
static void inverse_qft(quantumState_SV<P> &qs, const std::vector<int> &phase)
{
    const int p = (int)phase.size();
    // Standard in-place inverse QFT (no swaps if you choose same ordering everywhere)
    for (int j = p - 1; j >= 0; --j)
    {
        for (int k = p - 1; k > j; --k)
        {
            // Controlled phase: |11> gets phase -pi/2^{k-j}
            const double theta = -M_PI / std::ldexp(1.0, k - j); // -pi / 2^(k-j)
            qs.RZ(/*theta=*/theta, /*targets=*/{phase[j]}, /*controls=*/{phase[k]});
        }
        qs.H({phase[j]});
    }
}

// Apply controlled-U^{2^k} with U = exp(+iA t0)  (so time = t0 * 2^k)
// We implement U via Chebyshev (t=-1 for your API to get +iA) or Taylor.
template <precision P>
static int controlled_exp_iA_pow2(
    quantumState_SV<P> &qs,
    int nQubits_total,
    const int *d_csrRowPtr,
    const int *d_csrColInd,
    const cuDoubleComplex *d_csrVal,
    int nnz,
    int order,
    int k,
    double t0,
    int control_qubit,
    const std::vector<int> &target_qubits,
    bool use_chebyshev)
{
    const double scale = std::ldexp(t0, k); // t0 * 2^k
    if (use_chebyshev)
    {
        // Chebyshev host API: t = -scale → exp(+i A * scale)
        return qs.applyMatrixExponential_chebyshev(
            d_csrRowPtr, d_csrColInd, d_csrVal, nnz, order,
            /*targets*/ target_qubits,
            /*controls*/ std::vector<int>{control_qubit},
            /*t*/ -scale);
    }
    else
    {
        throw std::runtime_error("Not supported");
    }
}

// Placeholder: coherent reciprocal rotation  |0>_anc → sqrt(1-C^2/φ^2)|0> + (C/φ)|1>
// driven by the phase register |φ~>. Replace with your QROM/piecewise polynomial.
template <precision P>
static void reciprocal_rotation_placeholder(
    quantumState_SV<P> &qs,
    int solution_ancilla,
    const std::vector<int> &phase_register,
    double C /* scaling constant, <= min|λ| to keep <=1 */)
{
    // ⚠️ This is a no-op. Plug your multi-controlled RY approximant here.
    (void)qs;
    (void)solution_ancilla;
    (void)phase_register;
    (void)C;
}

// Project on ancilla |1> (post-selection) in host memory: zero amplitudes with anc=0; renormalize.
// Return probability of |1>.
template <precision P>
static double project_ancilla_one(quantumState_SV<P> &qs, int anc_idx)
{
    auto sv = qs.getStateVector(); // span<complex_type>
    using complex_type = typename std::remove_reference_t<decltype(sv)>::value_type;
    double prob1 = 0.0;
    const size_t dim = sv.size();
    for (size_t i = 0; i < dim; ++i)
    {
        const bool anc1 = ((i >> anc_idx) & 1u) != 0u;
        double r = (double)reinterpret_cast<const cuDoubleComplex &>(sv[i]).x;
        double im = (double)reinterpret_cast<const cuDoubleComplex &>(sv[i]).y;
        double amp2 = r * r + im * im;
        if (anc1)
            prob1 += amp2;
    }
    double inv_norm = (prob1 > 0.0) ? 1.0 / std::sqrt(prob1) : 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        const bool anc1 = ((i >> anc_idx) & 1u) != 0u;
        cuDoubleComplex &z = reinterpret_cast<cuDoubleComplex &>(sv[i]);
        if (!anc1)
        {
            z = make_cuDoubleComplex(0.0, 0.0);
        }
        else
        {
            z = make_cuDoubleComplex(z.x * inv_norm, z.y * inv_norm);
        }
    }
    return prob1;
}

// ----------------------------- HHL main ------------------------------

template <precision P>
HHLBenchResult run_hhl_benchmark(
    quantumState_SV<P> &qs,
    // A in CSR (device pointers)
    const int *d_csrRowPtr,
    const int *d_csrColInd,
    const cuDoubleComplex *d_csrVal,
    int n_target_dim, // size of A (2^#target_qubits)
    int nnz,
    // register layout in qs:
    int phase_base,                        // first phase qubit index
    int pe_qubits,                         // p
    int solution_anc,                      // index of the |0> ancilla used for reciprocal & postselect
    const std::vector<int> &target_qubits, // contiguous or arbitrary, your kernels already support mapping
    // algorithm params
    const HHLParams &params = {})
{
    assert(pe_qubits == params.pe_qubits && "pe_qubits mismatch with params");
    HHLBenchResult out{};
    CudaTimer t_total, t_phase, t_iqft;

    // 0) Hadamards on the phase register
    const auto phase_reg = phase_range(phase_base, pe_qubits);
    for (int q : phase_reg)
        qs.H({q});

    // 1) Controlled-U^{2^k} blocks  (U = exp(+iA t0))
    t_total.tic();
    t_phase.tic();
    for (int k = 0; k < pe_qubits; ++k)
    {
        const int ctrl = phase_reg[k]; // LSB-first convention
        const int rc = controlled_exp_iA_pow2(
            qs, /*nQubits_total unused*/ 0,
            d_csrRowPtr, d_csrColInd, d_csrVal, nnz,
            params.order, k, params.t0, ctrl, target_qubits, params.use_chebyshev);
        if (rc != 0)
            throw std::runtime_error("controlled_exp_iA_pow2 failed");
    }
    out.ms_phase_estimate = t_phase.toc();

    // 2) Inverse QFT on phase register
    t_iqft.tic();
    inverse_qft(qs, phase_reg);
    out.ms_inv_qft = t_iqft.toc();

    // 3) Reciprocal rotation on solution ancilla (placeholder)
    reciprocal_rotation_placeholder(qs, solution_anc, phase_reg, /*C=*/0.5);

    // 4) Uncompute phase register if you encoded reciprocal coherently (not needed for placeholder)

    // 5) Post-select on |1> of solution ancilla (optional; we do it for a sanity metric)
    out.post_select_prob = project_ancilla_one(qs, solution_anc);

    out.ms_total = t_total.toc();
    return out;
}
