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
#include "../functionality/SquareNorm.hpp"
#include "../CuSparseControl/SparseDenseConvert.hpp"

#define DEBUG_HHL 1

// ===== Debug helpers: subspace dumps via accessor =====

static inline cuDoubleComplex Cplx(double r, double i=0.0) {
    return make_cuDoubleComplex(r,i);
}

// Print a subspace ordered by `readOrderingQubits`, with an optional mask on other qubits.
// Example: readOrdering={SYS...}, maskOrdering={phase..., anc}, maskBits={0...,1} to print "anc=1, phase=0...0".
template <precision P>
static void dump_subspace(
    const char* label,
    quantumState_SV<P>& qs,
    std::span<const int> readOrderingQubits,
    std::span<const int> maskOrderingQubits,
    std::span<const int> maskBitString,
    int max_to_print = -1)  // -1 = print all
{
    const std::size_t out_len = (readOrderingQubits.empty() ? 1 : (std::size_t(1) << readOrderingQubits.size()));
    std::vector<PRECISION_TYPE_COMPLEX(P)> buf(out_len);

    // Pull amplitudes from the device
    qs.accessor_get_by_qubits(readOrderingQubits, maskOrderingQubits, maskBitString, std::span<PRECISION_TYPE_COMPLEX(P)>(buf));

    // Compute norm^2 of this subspace (probability mass in the mask)
    long double p = 0.0L;
    for (auto z : buf) {
        const long double re = cuCreal(z);
        const long double im = cuCimag(z);
        p += re*re + im*im;
    }

    std::cout << "\n--- " << label << " ---\n";
    std::cout << "subspace size = " << buf.size() << ", mass = " << static_cast<double>(p) << "\n";
    std::cout.setf(std::ios::fixed); std::cout.precision(10);

    const std::size_t limit = (max_to_print < 0) ? buf.size() : std::min<std::size_t>(buf.size(), max_to_print);
    for (std::size_t i = 0; i < limit; ++i) {
        std::cout << "amp[" << i << "] = (" << cuCreal(buf[i]) << ", " << cuCimag(buf[i]) << ")\n";
    }
    if (limit < buf.size()) std::cout << "... (" << (buf.size() - limit) << " more)\n";
}

// For tiny systems only: dump full |ψ⟩ in computational order [q=0 is LSB].
template <precision P>
static void dump_full_state_small(quantumState_SV<P>& qs, int total_qubits, int max_to_print = -1)
{
    std::vector<int> order(total_qubits);
    for (int q=0;q<total_qubits;++q) order[q]=q;

    const std::size_t N = (total_qubits == 0 ? 1 : (std::size_t(1) << total_qubits));
    std::vector<PRECISION_TYPE_COMPLEX(P)> buf(N);

    qs.accessor_get_by_qubits(order, /*maskOrder*/{}, /*maskBits*/{}, buf);

    long double norm2 = 0.0L;
    for (auto z : buf) { long double re=cuCreal(z), im=cuCimag(z); norm2 += re*re+im*im; }

    std::cout << "\n=== Full state dump ("<< N <<" amps), ||ψ||^2=" << static_cast<double>(norm2) << " ===\n";
    const std::size_t limit = (max_to_print < 0) ? N : std::min<std::size_t>(N, max_to_print);
    std::cout.setf(std::ios::fixed); std::cout.precision(10);
    for (std::size_t i=0;i<limit;++i) {
        std::cout << "ψ["<< i << "] = (" << cuCreal(buf[i]) << ", " << cuCimag(buf[i]) << ")\n";
    }
    if (limit < N) std::cout << "... (" << (N - limit) << " more)\n";
}


struct HHL_options
{
    int num_phaseReg = 4;
    int num_SYSReg = 3;
    int num_ancilla = 1;
    int t0 = 1;
    std::vector<cuDoubleComplex> b;
    std::vector<cuDoubleComplex> A;
};
int HHL_run(HHL_options options)
{
    const int p   = options.num_phaseReg;
    const int ns  = options.num_SYSReg;
    const int na  = options.num_ancilla;
    const int total_qubits = p + ns + na;

    // layout: phase [0..p-1], system [p..p+ns-1], ancilla [p+ns]
    int cur = 0;
    std::vector<int> phase_qubits(p);  for (int i=0;i<p;++i)  phase_qubits[i] = cur++;
    std::vector<int> system_qubits(ns);for (int i=0;i<ns;++i) system_qubits[i]= cur++;
    std::vector<int> ancilla_qubit(na);for (int i=0;i<na;++i) ancilla_qubit[i]= cur++;
    const int anc = ancilla_qubit[0];

    // 1. Not Phase Register (System + Ancilla)
    std::vector<int> not_phase_qubits(ns + na);
    // Start after the phase register (index 'p')
    // and go until the end (index 'total_qubits')
    for (int i = 0; i < ns + na; ++i) {
        not_phase_qubits[i] = p + i;
    }

    // 2. Not System Register (Phase + Ancilla)
    std::vector<int> not_system_qubits(p + na);
    
    // a) Add Phase Qubits (0 to p-1)
    for (int i = 0; i < p; ++i) {
        not_system_qubits[i] = i;
    }
    // b) Add Ancilla Qubits (p + ns to total_qubits - 1)
    for (int i = 0; i < na; ++i) {
        not_system_qubits[p + i] = p + ns + i;
    }

    // 3. Not Ancilla Register (Phase + System)
    std::vector<int> not_ancilla_qubits(p + ns);
    // These are the first p + ns qubits (0 to p+ns-1)
    for (int i = 0; i < p + ns; ++i) {
        not_ancilla_qubits[i] = i;
    }

    // CSR(A)
    const int n = 1 << ns;
    std::vector<int> row_ptr_A, col_ind_A;
    std::vector<cuDoubleComplex> values_A;
    dense_to_csr(std::span<const cuDoubleComplex>(options.A), n, row_ptr_A, col_ind_A, values_A, 0.0);

    quantumState_SV<precision::bit_64> qs(total_qubits);
    qs.prefetchToDevice();

    // Write |b> into the system subspace across *all* non-target blocks (simple for debugging).
    qs.write_amplitudes_to_target_qubits(std::span<const cuDoubleComplex>(options.b),
                                         std::span<const int>(system_qubits));

#if DEBUG_HHL
    dump_subspace("After |b> load (system only, no mask)", qs,
                  std::span<const int>(system_qubits),
                  /*maskOrder*/std::vector{0,1,2,3,7}, /*maskBits*/std::vector{0,0,0,0,0}, /*max_to_print*/ (1<<std::min(6,ns)));
    // Uncomment if system is tiny:
    // dump_full_state_small(qs, total_qubits, 64);
#endif
    // Hadamards on phase
    qs.H(std::span<const int>(phase_qubits));
#if DEBUG_HHL
    dump_subspace("After H on phase (phase register only)", qs,
                  std::span<const int>(phase_qubits),
                  /*maskOrder*/{}, /*maskBits*/{}, /*max*/ (1<<std::min(8,p)));
    // Show system while fixing anc=0 (so slices are interpretable)
    {
        std::vector<int> maskOrder = { anc };
        std::vector<int> maskBits  = { 0   };
        dump_subspace("System slice with anc=0 (phase free)", qs,
                      std::span<const int>(system_qubits), maskOrder, maskBits, (1<<std::min(6,ns)));
    }
#endif

    // Controlled U^{2^k}: Chebyshev with t = -(t0 * 2^k) → e^{+i A t0 2^k}
    constexpr int ORDER = 30;
    for (int k = 0; k < p; ++k)
    {
        const int ctrlQ = phase_qubits[k];
        const double tk = -static_cast<double>(options.t0) * pow(2,k);
        qs.applyMatrixExponential_chebyshev(
            row_ptr_A.data(), col_ind_A.data(), values_A.data(),
            values_A.size(),
            ORDER,
            system_qubits,
            {ctrlQ},
            tk);

#if DEBUG_HHL
        // peek: system subspace with (for readability) anc=0
        std::vector<int> mO = { anc }, mB = { 0 };
        char label[128];
        std::snprintf(label, sizeof(label), "After controlled exp step k=%d", k);
        dump_subspace(label, qs, std::span<const int>(system_qubits), mO, mB, (1<<std::min(6,ns)));
#endif
    }

    // Inverse QFT on phase
    for (int j = p-1; j >= 0; --j) {
        const int tgt = phase_qubits[j];
        for (int k = p-1; k > j; --k) {
            const int ctrl = phase_qubits[k];
            const double theta = -M_PI / pow(2,k-j);
            qs.RZ(theta, {tgt}, {ctrl});
        }
        qs.H({tgt});
    }
#if DEBUG_HHL
    dump_subspace("After IQFT (phase only)", qs,
                  std::span<const int>(phase_qubits), {}, {}, (1<<std::min(8,p)));
#endif
// Controlled ancilla rotations (bitstring-conditioned on phase)
// This section implements the core CNOT-like-R_y operation in HHL,
// rotating the ancilla qubit based on the eigenvalue (phase) estimate.

// 1. Determine the scaling factor for the eigenvalue estimate.
const double two_pi_over_t0 = (2.0 * M_PI) / static_cast<double>(options.t0);

// 2. Find the minimum non-zero eigenvalue estimate (lambda_hat_min).
// This minimum is used to calculate the scaling constant C.
double min_lambda_hat = std::numeric_limits<double>::infinity();
// Loop over all possible states 's' (bitstrings) in the phase register (from 1 to 2^p - 1)
for (std::uint64_t s=1; s < (1ull<<p); ++s) {
    const double phi = double(s) / double(1ull<<p);             // Phase fraction: phi = s / 2^p
    const double lambda_hat = two_pi_over_t0 * phi;             // Eigenvalue estimate: lambda_hat = (2*pi/t0) * phi
    if (lambda_hat > 0.0) min_lambda_hat = std::min(min_lambda_hat, lambda_hat);
}

// 3. Apply the controlled-Ry rotations for all non-zero eigenvalues.
if (std::isfinite(min_lambda_hat) && min_lambda_hat > 0.0) {
    // Scaling constant C = 0.5 * lambda_hat_min. This ensures C / lambda_hat <= 1.0.
    const double C = 0.5 * min_lambda_hat;
    
    std::vector<int> flip_bits; flip_bits.reserve(p);
    
    // Loop over all possible states 's' (bitstrings) in the phase register (from 0 to 2^p - 1)
    for (std::uint64_t s=0; s < (1ull<<p); ++s) {
        flip_bits.clear();
        
        // Identify which control bits are '0' for state 's'.
        // These will need to be flipped to '1' using X gates for the Controlled-Ry.
        for (int qpos=0; qpos<p; ++qpos)
            if (((s>>qpos)&1ull)==0ull) flip_bits.push_back(phase_qubits[qpos]);

        const double phi = double(s)/double(1ull<<p);
        const double lambda_hat = two_pi_over_t0 * phi; // Eigenvalue estimate
        
        // Skip states corresponding to lambda <= 0.
        if (lambda_hat <= 0.0) continue;

        // Calculate the rotation parameter 'x' (proportional to 1/lambda_hat).
        // This is the desired sin(theta/2) value for the rotation.
        double x = C / lambda_hat;
        
        // Clamp x to [0, 1] to prevent floating point errors or non-physical arcsin arguments.
        if (x > 1.0) x = 1.0;
        if (x < 0.0) x = 0.0;
        
        // Calculate the final rotation angle theta = 2 * arcsin(x).
        // The Ry(theta) gate will rotate the ancilla to a state where P(ancilla=1) = x^2.
        const double theta = 2.0 * std::asin(x);
        if (theta == 0.0) continue; // Skip if no rotation is required

        // 1. Pre-flip the '0' bits to '1' (so the Controlled-Ry targets state 's').
        if (!flip_bits.empty()) qs.X(std::span<const int>(flip_bits));
        
        // 2. Apply the Controlled-Ry(theta) gate.
        // Target: anc, Controls: phase_qubits (all of them).
        qs.RY(theta, std::span<const int>(&anc,1), std::span<const int>(phase_qubits));
        
        // 3. Un-flip the bits to restore the phase register to its original state.
        if (!flip_bits.empty()) qs.X(std::span<const int>(flip_bits));
    }
}

#if DEBUG_HHL
    // look at ancilla marginal (just the ancilla qubit amplitudes)
    dump_subspace("Ancilla marginal (anc only)", qs,
                  std::span<const int>(ancilla_qubit), {}, {}, 2);
    // system conditioned on anc=1, phase free:
    { std::vector<int> mO = { anc }; std::vector<int> mB = { 1 };
      dump_subspace("System with anc=1 (phase free)", qs,
                    std::span<const int>(system_qubits), mO, mB, (1<<std::min(6,ns))); }
#endif

    // UNCOMPUTE: forward QFT then inverse evolutions
    for (int j = 0; j < p; ++j) {
        const int tgt = phase_qubits[j];
        qs.H(std::span<const int>(&tgt,1));
        for (int k = j+1; k < p; ++k) {
            const int ctrl = phase_qubits[k];
            const double theta = +M_PI / static_cast<double>(1 << (k-j));
            qs.RZ(theta, std::span<const int>(&tgt,1), std::span<const int>(&ctrl,1));
        }
    }
    for (int k = p-1; k >= 0; --k) {
        const int ctrlQ = phase_qubits[k];
        const double tk = +static_cast<double>(options.t0) * static_cast<double>(1ull << k);
        qs.applyMatrixExponential_chebyshev(
            row_ptr_A.data(), col_ind_A.data(), values_A.data(),
            static_cast<int>(values_A.size()),
            ORDER,
            system_qubits,
            std::vector<int>{ctrlQ},
            tk);
    }

#if DEBUG_HHL
    // PE should be back to |0...0>, inspect that marginal
    dump_subspace("After uncompute: phase marginal", qs,
                  std::span<const int>(phase_qubits), {}, {}, (1<<std::min(8,p)));
#endif

    // FINAL: system slice with anc=1, phase=0...0 (what you asked to print earlier)
    std::vector<int> mask_order; std::vector<int> mask_bits;
    mask_order.reserve(p+1); mask_bits.reserve(p+1);
    for (int q : phase_qubits) { mask_order.push_back(q); mask_bits.push_back(0); }
    mask_order.push_back(anc);  mask_bits.push_back(1);

    dump_subspace("FINAL: system amplitudes |anc=1, phase=0...0>", qs,
                  std::span<const int>(system_qubits),
                  std::span<const int>(mask_order),
                  std::span<const int>(mask_bits),
                  (1<<std::min(12,ns))); // print all if ns<=12

    return cudaSuccess;
}
