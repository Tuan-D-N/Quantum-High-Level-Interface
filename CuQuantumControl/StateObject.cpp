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

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::applyArbitaryGateUnsafe(std::span<const int> targets, std::span<const int> controls, std::span<const complex_t> matrix)
{
    THROW_BROAD_ERROR(applyGatesGeneral<selectedPrecision>(m_handle,
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
    THROW_CUSTATEVECTOR(custatevecCreate(&m_handle));
    setNumberOfQubits(nQubits);
}

template <precision selectedPrecision>
quantumState_SV<selectedPrecision>::quantumState_SV(std::span<const complex_t> stateVector)
{
    THROW_CUSTATEVECTOR(custatevecDestroy(m_handle));
    setStateVector(stateVector);
}

template <precision selectedPrecision>
quantumState_SV<selectedPrecision>::~quantumState_SV()
{
    freeWorkspace();
    freeStateVector();
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::setStateVector(std::span<const complex_t> stateVector)
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

    THROW_CUDA(cudaMallocManaged((void **)&m_stateVector, nSV * sizeof(complex_t)));
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
    cudaMemPrefetchAsync(m_stateVector, nSV * sizeof(complex_t), deviceNumber);
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::prefetchToCPU()
{
    size_t nSV = 1 << m_numberQubits;
    cudaMemPrefetchAsync(m_stateVector, nSV * sizeof(complex_t), cudaCpuDeviceId);
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::applyArbitaryGate(std::span<const int> targets, std::span<const int> controls, std::span<const complex_t> matrix)
{
    if (targets.size() <= 0)
    {
        throw std::logic_error("zero targets to run");
    }
    if (!are_disjoint(targets, controls))
    {
        throw std::logic_error("targets and controls are not independent set");
    }
    if (matrix.size() == std::pow((1 << targets.size()), 2))
    {
        throw std::logic_error("matrix size is missmatch with target's size");
    }
    applyArbitaryGateUnsafe(targets, controls, matrix);
}

template <precision selectedPrecision>
void quantumState_SV<selectedPrecision>::applyArbitaryGate(std::initializer_list<const int> targets, std::initializer_list<const int> controls, std::initializer_list<const complex_t> matrix)
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
            m_handle,                                                                                        \
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
            m_handle,                                                                                        \
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
