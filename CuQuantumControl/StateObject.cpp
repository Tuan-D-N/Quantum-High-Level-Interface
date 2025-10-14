#include <custatevec.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <iostream>
#include <span>
#include <vector>
#include <cassert>
#include <initializer_list>

#include "../CudaControl/Helper.hpp"
#include "../functionality/FUNCTION_MACRO_UTIL.hpp"
#include "ApplyGates.hpp"
#include "MatriceDefinitions.hpp"
#include "Precision.hpp"
#include "StateObject.hpp"

// ===================== Implementation =====================
// Low-level: directly forwards to your applyAccessorGet template
template <precision selectedPrecision>
int quantumState_SV<selectedPrecision>::accessor_get_raw(
    int nIndexBits,
    std::span<const int> bitOrdering,
    std::span<const int> maskBitString,
    std::span<const int> maskOrdering,
    int buffer_access_begin,
    int buffer_access_end,
    std::span<PRECISION_TYPE_COMPLEX(selectedPrecision)> out_buffer)
{
    static_assert(selectedPrecision == precision::bit_32 || selectedPrecision == precision::bit_64,
                  "Unsupported precision");

    // Basic sanity
    assert(m_stateVector != nullptr);
    assert(m_custatevec_handle != nullptr);
    assert(nIndexBits >= 0);
    assert(bitOrdering.size() == static_cast<std::size_t>(nIndexBits));
    assert(maskBitString.size() == maskOrdering.size()); // both 0 means no mask
    assert(buffer_access_begin >= 0 && buffer_access_end >= buffer_access_begin);

    // custatevec requires raw pointers
    const int *bitOrderingPtr = bitOrdering.empty() ? nullptr : bitOrdering.data();
    const int *maskBitStringPtr = maskBitString.empty() ? nullptr : maskBitString.data();
    const int *maskOrderingPtr = maskOrdering.empty() ? nullptr : maskOrdering.data();

    // Call your provided function template
    return applyAccessorGet<selectedPrecision>(
        m_custatevec_handle,
        static_cast<int>(m_numberQubits), // nIndexBits MUST be total index bits of the full state
        static_cast<int>(bitOrdering.size()),
        bitOrderingPtr,
        static_cast<int>(maskBitString.size()),
        maskBitStringPtr,
        maskOrderingPtr,
        m_stateVector,
        buffer_access_begin,
        buffer_access_end,
        out_buffer.data(),
        m_extraWorkspace,
        m_extraWorkspaceSizeInBytes);
}

template <precision selectedPrecision>
int quantumState_SV<selectedPrecision>::accessor_get_raw(
    std::span<const int> bitOrdering,
    std::span<const int> maskBitString,
    std::span<const int> maskOrdering,
    std::span<PRECISION_TYPE_COMPLEX(selectedPrecision)> out_buffer)
{
    static_assert(selectedPrecision == precision::bit_32 || selectedPrecision == precision::bit_64,
                  "Unsupported precision.");
    assert(m_stateVector != nullptr);
    assert(m_custatevec_handle != nullptr);

    // Validate masks
    if (maskBitString.size() != maskOrdering.size())
        throw std::invalid_argument("maskBitString and maskOrdering must have the same length.");

    // Expected full-subspace length = 2^{|bitOrdering|}
    const std::size_t expected_len =
        (bitOrdering.size() <= static_cast<std::size_t>(sizeof(std::size_t) * 8))
            ? (std::size_t(1) << bitOrdering.size())
            : 0; // overflow guard; practically unreachable for state sizes we can hold

    if (expected_len == 0)
        throw std::overflow_error("bitOrdering too large for subspace size computation.");

    if (out_buffer.size() != expected_len)
        throw std::invalid_argument("out_buffer.size() must equal 2^{|bitOrdering|} for auto-range accessor.");

    // Auto range: [0, expected_len)
    const int buffer_access_begin = 0;
    const int buffer_access_end = static_cast<int>(expected_len);

    // Forward to your existing accessor (range-explicit)
    return accessor_get_raw(
        static_cast<int>(m_numberQubits),
        bitOrdering,
        maskBitString,
        maskOrdering,
        buffer_access_begin,
        buffer_access_end,
        out_buffer);
}

template <precision selectedPrecision>
int quantumState_SV<selectedPrecision>::accessor_get_by_qubits(
    std::span<const int> readOrderingQubits,
    std::span<const int> maskOrderingQubits,
    std::span<const int> maskBitString,
    std::span<PRECISION_TYPE_COMPLEX(selectedPrecision)> out_buffer)
{
    if (maskOrderingQubits.size() != maskBitString.size())
        throw std::invalid_argument("maskBitString and maskOrderingQubits must have the same length.");

    // Build contiguous arrays custatevec expects
    std::vector<int> bitOrderingVec(readOrderingQubits.begin(), readOrderingQubits.end());
    std::vector<int> maskOrderingVec(maskOrderingQubits.begin(), maskOrderingQubits.end());
    std::vector<int> maskBitsVec(maskBitString.begin(), maskBitString.end());

    // Bounds checks
    for (int q : bitOrderingVec)
        if (q < 0 || q >= static_cast<int>(m_numberQubits))
            throw std::out_of_range("readOrderingQubits contains an out-of-range qubit.");

    for (int q : maskOrderingVec)
        if (q < 0 || q >= static_cast<int>(m_numberQubits))
            throw std::out_of_range("maskOrderingQubits contains an out-of-range qubit.");

    for (int b : maskBitsVec)
        if (b != 0 && b != 1)
            throw std::invalid_argument("maskBitString must contain only 0/1 values.");

    // Expected full-subspace size = 2^{|readOrderingQubits|}
    const std::size_t expected_len = (bitOrderingVec.size() <= static_cast<std::size_t>(sizeof(std::size_t) * 8))
                                         ? (std::size_t(1) << bitOrderingVec.size())
                                         : 0;

    if (expected_len == 0)
        throw std::overflow_error("readOrderingQubits too large for subspace size computation.");

    if (out_buffer.size() != expected_len)
        throw std::invalid_argument("out_buffer.size() must equal 2^{|readOrderingQubits|} for auto-range accessor.");

    // Auto range: [0, 2^{|readOrderingQubits|})
    const int buffer_access_begin = 0;
    const int buffer_access_end = static_cast<int>(expected_len);

    return accessor_get_raw(
        static_cast<int>(m_numberQubits),
        std::span<const int>(bitOrderingVec),
        std::span<const int>(maskBitsVec),
        std::span<const int>(maskOrderingVec),
        buffer_access_begin,
        buffer_access_end,
        out_buffer);
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::write_amplitudes_to_target_qubits(
    std::span<const cuDoubleComplex> amplitudes_b,
    std::span<const std::uint64_t> targetQubits)
{
    // This helper is implemented for double-precision statevectors only.
    if constexpr (!std::is_same_v<complex_type, cuDoubleComplex>)
    {
        throw std::runtime_error(
            "write_amplitudes_to_target_qubits requires cuDoubleComplex (64-bit complex).");
    }
    else
    {
        assert(m_stateVector != nullptr);
        // nQubits fits in uint64_t naturally
        const std::uint64_t nQubitsTotal = static_cast<std::uint64_t>(m_numberQubits);

        // Basic sanity: target indices must be within [0, nQubitsTotal)
        for (std::uint64_t q : targetQubits)
        {
            if (!(q < nQubitsTotal))
            {
                throw std::out_of_range("Target qubit index out of range.");
            }
        }

        write_amplitudes_to_target_qubits_u64(
            m_stateVector,
            nQubitsTotal,
            amplitudes_b,
            targetQubits);
    }
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::write_amplitudes_to_target_qubits(
    std::span<const cuDoubleComplex> amplitudes_b,
    std::span<const int> targetQubits)
{
    // Convert int -> uint64_t without C-style casts
    std::vector<std::uint64_t> tgt;
    tgt.reserve(static_cast<std::size_t>(targetQubits.size()));
    for (int q : targetQubits)
    {
        if (q < 0)
        {
            throw std::out_of_range("Target qubit index must be non-negative.");
        }
        tgt.push_back(static_cast<std::uint64_t>(q));
    }
    // Forward to the 64-bit overload
    write_amplitudes_to_target_qubits(amplitudes_b, std::span<const std::uint64_t>(tgt));
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::zero_state()
{
    assert(m_stateVector != nullptr);
    THROW_CUDA(cudaMemset(m_stateVector, 0, 1 << m_numberQubits));
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::normalise_SV()
{
    assert(m_cusparse_handle != nullptr);
    assert(m_stateVector != nullptr);

    if constexpr (std::is_same_v<complex_type, cuDoubleComplex>)
    {
        size_t length = 1ull << m_numberQubits;
        THROW_CUDA(square_normalize_statevector_u64(m_stateVector, length, nullptr));
    }
    else
    {
        throw std::runtime_error("Error: get_state_span is not implemented/supported for single precision (cuComplex).");
    }
}

// =============================================================
// Apply a sparse CSR matrix directly to m_stateVector (in place)
// =============================================================
template <precision selectedPrecision>
int quantumState_SV<selectedPrecision>::applySparseMatrixStateVectorWide(
    std::span<int> csrOffsets,
    std::span<int> csrColumns,
    std::span<cuDoubleComplex> csrValues)
{
    assert(m_cusparse_handle != nullptr);
    assert(m_stateVector != nullptr);

    if constexpr (std::is_same_v<complex_type, cuDoubleComplex>)
    {
        // Allocate a temporary device buffer for intermediate output
        size_t dim = 1ull << m_numberQubits;
        cuDoubleComplex *d_temp = nullptr;
        CHECK_CUDA(cudaMalloc(&d_temp, dim * sizeof(cuDoubleComplex)));

        // Perform sparse multiplication
        int status = applySparseCSRMat(
            m_cusparse_handle,
            csrOffsets,
            csrColumns,
            csrValues,
            std::span<cuDoubleComplex>(m_stateVector, dim),
            std::span<cuDoubleComplex>(d_temp, dim));

        CHECK_BROAD_ERROR(status);

        // Overwrite m_stateVector with the result
        CHECK_CUDA(cudaMemcpy(m_stateVector, d_temp,
                              dim * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToDevice));

        CHECK_CUDA(cudaFree(d_temp));
        return status;
    }
    else
    {
        throw std::runtime_error("Error: get_state_span is not implemented/supported for single precision (cuComplex).");
    }
}
template <precision selectedPrecision>
int quantumState_SV<selectedPrecision>::applySparseGateToSV(
    std::span<const int> d_csrRowPtrU,
    std::span<const int> d_csrColIndU,
    std::span<const cuDoubleComplex> d_csrValU,
    const std::vector<int> &targetQubits,
    const std::vector<int> &controlQubits)
{
    // Enforce cuDoubleComplex precision (return runtime error code if not supported)
    if constexpr (std::is_same_v<complex_type, cuDoubleComplex>)
    {

        // Basic state sanity
        if (!m_stateVector)
            return -10; // state not initialized
        if (m_numberQubits == 0)
            return -11; // invalid qubit count

        // Dimension N = 2^{m_numberQubits}
        const size_t Nsz = size_t{1} << m_numberQubits;
        if (Nsz > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            // Requires 64-bit CSR indices; this path is 32-bit only.
            return -12;
        }
        const int N = static_cast<int>(Nsz);

        // --- Basic checks on the 3 spans (sizes only; do NOT dereference device memory) ---
        if (d_csrColIndU.size() != d_csrValU.size())
            return -21;
        const int nnzU = static_cast<int>(d_csrColIndU.size());
        if (nnzU <= 0)
            return -22;

        {
            std::vector<int> tmp = targetQubits;
            std::sort(tmp.begin(), tmp.end());
            if (std::adjacent_find(tmp.begin(), tmp.end()) != tmp.end())
                return -32; // duplicate targets
        }
        {
            std::vector<int> tmp = controlQubits;
            std::sort(tmp.begin(), tmp.end());
            if (std::adjacent_find(tmp.begin(), tmp.end()) != tmp.end())
                return -33; // duplicate controls
        }
        {
            // Check disjointness (not strictly required if CU already encodes logic, but good hygiene)
            std::vector<int> t = targetQubits, c = controlQubits;
            std::vector<int> inter;
            std::set_intersection(t.begin(), t.end(), c.begin(), c.end(), std::back_inserter(inter));
            if (!inter.empty())
                return -34; // overlap between targets and controls
        }

        // --- Allocate temporary output on device ---
        cuDoubleComplex *d_state_out = nullptr;
        CHECK_CUDA(cudaMalloc(&d_state_out, N * sizeof(cuDoubleComplex)));

        applySparseGate(m_cusparse_handle,
                        static_cast<int>(m_numberQubits),
                        d_csrRowPtrU.data(),
                        d_csrColIndU.data(),
                        d_csrValU.data(),
                        m_stateVector,
                        d_state_out,
                        targetQubits,
                        controlQubits,
                        nnzU);

        // --- Copy result back into member state and free temp ---
        CHECK_CUDA(cudaMemcpy(
            m_stateVector,
            d_state_out,
            N * sizeof(cuDoubleComplex),
            cudaMemcpyDeviceToDevice));

        cudaFree(d_state_out);
        return cudaSuccess; // or 0
    }
    else
    {
        throw std::runtime_error("Error: get_state_span is not implemented/supported for single precision (cuComplex).");
        return -999; // NOT_IMPLEMENTED_FOR_THIS_PRECISION
    }
}
// =============================================================
// Apply e^{iA} (using truncated Taylor series) to the statevector
// =============================================================
template <precision selectedPrecision>
int quantumState_SV<selectedPrecision>::applyMatrixExponential_taylor(
    const int *d_csrRowPtr,
    const int *d_csrColInd,
    const cuDoubleComplex *d_csrVal,
    int nnz,
    int order,
    const std::vector<int> &targetQubits,
    const std::vector<int> &controlQubits)
{
    if constexpr (std::is_same_v<complex_type, cuDoubleComplex>)
    {
        assert(m_cusparse_handle != nullptr);
        assert(m_stateVector != nullptr);

        return applyControlledExpTaylor_cusparse(
            m_cusparse_handle,
            static_cast<int>(m_numberQubits),
            d_csrRowPtr,
            d_csrColInd,
            d_csrVal,
            m_stateVector,
            targetQubits,
            controlQubits,
            nnz,
            order);
    }
    else
    {
        throw std::runtime_error("Error: get_state_span is not implemented/supported for single precision (cuComplex).");
    }
}

template <precision selectedPrecision>
int quantumState_SV<selectedPrecision>::applyMatrixExponential_chebyshev(
    const int *d_csrRowPtr,
    const int *d_csrColInd,
    const cuDoubleComplex *d_csrVal,
    int nnz,
    int order,
    const std::vector<int> &targetQubits,
    const std::vector<int> &controlQubits,
    double t /* = +1.0 for exp(-iA), -1.0 for exp(+iA) */)
{
    if constexpr (std::is_same_v<complex_type, cuDoubleComplex>)
    {
        assert(m_cusparse_handle != nullptr);
        assert(m_stateVector != nullptr);

        return applyControlledExpChebyshev_cusparse_host(
            m_cusparse_handle,
            static_cast<int>(m_numberQubits),
            d_csrRowPtr,
            d_csrColInd,
            d_csrVal,
            m_stateVector,
            targetQubits,
            controlQubits,
            nnz,
            order,
            t);
    }
    else
    {
        throw std::runtime_error(
            "Error: applyMatrixExponential_chebyshev is not implemented/supported for single precision (cuComplex).");
    }
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::applyArbitaryGateUnsafe(std::span<const int> targets, std::span<const int> controls, std::span<const complex_type> matrix)
{
    THROW_BROAD_ERROR(applyGatesGeneral<selectedPrecision>(m_custatevec_handle,
                                                           m_numberQubits,
                                                           matrix,
                                                           m_adjoint,
                                                           targets,
                                                           controls,
                                                           m_stateVector,
                                                           m_extraWorkspace,
                                                           m_extraWorkspaceSizeInBytes));
}

template <precision selectedPrecision>
quantumState_SV<selectedPrecision>::quantumState_SV(size_t nQubits)
{
    THROW_CUSTATEVECTOR(custatevecCreate(&m_custatevec_handle));
    THROW_CUSPARSE(cusparseCreate(&m_cusparse_handle));
    setNumberOfQubits(nQubits);
}

template <precision selectedPrecision>
quantumState_SV<selectedPrecision>::quantumState_SV(std::span<const complex_type> stateVector)
{
    THROW_CUSTATEVECTOR(custatevecCreate(&m_custatevec_handle));
    THROW_CUSPARSE(cusparseCreate(&m_cusparse_handle));
    setStateVector(stateVector);
}

template <precision selectedPrecision>
quantumState_SV<selectedPrecision>::~quantumState_SV()
{
    THROW_CUSTATEVECTOR(custatevecDestroy(m_custatevec_handle));
    THROW_CUSPARSE(cusparseDestroy(m_cusparse_handle));
    freeWorkspace();
    freeStateVector();
}

template <precision selectedPrecision>
std::span<PRECISION_TYPE_COMPLEX(selectedPrecision)> quantumState_SV<selectedPrecision>::getStateVector()
{
    return std::span<complex_type>(m_stateVector, m_numberQubits);
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::setStateVector(std::span<const complex_type> stateVector)
{
    if (!isPowerOf2(stateVector.size()))
    {
        throw std::runtime_error("Statevector is not a power of 2");
    }

    size_t numberOfQubits = std::log2(stateVector.size());

    setNumberOfQubits(numberOfQubits);
    for (int i = 0; i < stateVector.size(); ++i)
    {
        m_stateVector[i] = stateVector[i];
    }
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::setNumberOfQubits(size_t nQubits)
{
    if (nQubits <= 0)
    {
        throw std::logic_error("nQubits is less or equal to 0. Needs to be a natural number");
    }
    if (m_stateVector != nullptr)
    {
        freeStateVector();
    }

    size_t nSV = 1 << nQubits;

    THROW_CUDA(cudaMallocManaged((void **)&m_stateVector, nSV * sizeof(complex_type)));
    zero_state();
    m_numberQubits = nQubits;
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::freeWorkspace()
{
    if (m_extraWorkspace != nullptr)
    {
        THROW_CUDA(cudaFree(m_extraWorkspace));
        m_extraWorkspaceSizeInBytes = 0;
        m_extraWorkspace = nullptr;
    }
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::freeStateVector()
{
    if (m_stateVector != nullptr)
    {
        THROW_CUDA(cudaFree(m_stateVector));
        m_numberQubits = 0;
        m_stateVector = nullptr;
    }
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::prefetchToDevice(int deviceNumber)
{
    size_t nSV = 1 << m_numberQubits;
    cudaMemPrefetchAsync(m_stateVector, nSV * sizeof(complex_type), deviceNumber);
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::prefetchToCPU()
{
    size_t nSV = 1 << m_numberQubits;
    cudaMemPrefetchAsync(m_stateVector, nSV * sizeof(complex_type), cudaCpuDeviceId);
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::applyArbitaryGate(std::span<const int> targets, std::span<const int> controls, std::span<const complex_type> matrix)
{
    if (targets.size() <= 0)
    {
        throw std::logic_error("zero targets to run");
    }
    if (!are_disjoint(targets, controls))
    {
        throw std::logic_error("targets and controls are not independent set");
    }
    if (matrix.size() != std::pow((1 << targets.size()), 2))
    {
        throw std::logic_error("matrix size is missmatch with target's size");
    }
    applyArbitaryGateUnsafe(targets, controls, matrix);
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::applyArbitaryGate(std::initializer_list<const int> targets, std::initializer_list<const int> controls, std::initializer_list<const complex_type> matrix)
{
    // always remember that there is an alternative function of this that could cause this to be called recursively.
    applyArbitaryGate(std::span(targets),
                      std::span(controls),
                      std::span(matrix));
}

#define MAKE_GATES_BACKEND(GATE_NAME, NUMBER_OF_EXTRA_PARAMS)                                                \
    template <precision selectedPrecision>                                                                   \
    void quantumState_SV<selectedPrecision>::GATE_NAME(________SELECT_EXTRA_ARGS_PRE(NUMBER_OF_EXTRA_PARAMS) \
                                                           std::span<const int>                              \
                                                               targets,                                      \
                                                       std::span<const int>                                  \
                                                           controls)                                         \
    {                                                                                                        \
        THROW_BROAD_ERROR(apply##GATE_NAME<selectedPrecision>(                                               \
            m_custatevec_handle,                                                                             \
            m_numberQubits,                                                                                  \
            m_adjoint,                                                                                       \
            targets,                                                                                         \
            m_stateVector,                                                                                   \
            m_extraWorkspace,                                                                                \
            m_extraWorkspaceSizeInBytes                                                                      \
                ________SELECT_EXTRA_VARS_POST(NUMBER_OF_EXTRA_PARAMS)));                                    \
    }                                                                                                        \
                                                                                                             \
    template <precision selectedPrecision>                                                                   \
    void quantumState_SV<selectedPrecision>::GATE_NAME(________SELECT_EXTRA_ARGS_PRE(NUMBER_OF_EXTRA_PARAMS) \
                                                           std::initializer_list<const int>                  \
                                                               targets,                                      \
                                                       std::initializer_list<const int>                      \
                                                           controls)                                         \
    {                                                                                                        \
        THROW_BROAD_ERROR(apply##GATE_NAME<selectedPrecision>(                                               \
            m_custatevec_handle,                                                                             \
            m_numberQubits,                                                                                  \
            m_adjoint,                                                                                       \
            targets,                                                                                         \
            m_stateVector,                                                                                   \
            m_extraWorkspace,                                                                                \
            m_extraWorkspaceSizeInBytes                                                                      \
                ________SELECT_EXTRA_VARS_POST(NUMBER_OF_EXTRA_PARAMS)));                                    \
    }

MAKE_GATES_BACKEND(X, 0)
MAKE_GATES_BACKEND(Y, 0)
MAKE_GATES_BACKEND(Z, 0)
MAKE_GATES_BACKEND(H, 0)

MAKE_GATES_BACKEND(RX, 1)
MAKE_GATES_BACKEND(RY, 1)
MAKE_GATES_BACKEND(RZ, 1)
