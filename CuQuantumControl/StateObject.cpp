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
void quantumState_SV<selectedPrecision>::setStateVector(std::span<complex_t> stateVector)
{
    if (!isPowerOf2(stateVector))
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

#define MAKE_GATES_BACKEND(GATE_NAME, NUMBER_OF_EXTRA_PARAMS)                                                 \
    template <precision selectedPrecision>                                                                    \
    void quantumState_SV<selectedPrecision>::GATE_NAME(std::span<const int> targets                           \
                                                           ________SELECT_EXTRA_ARGS(NUMBER_OF_EXTRA_PARAMS), \
                                                       std::span<const int> controls)                         \
    {                                                                                                         \
        apply##GATE_NAME<selectedPrecision>(                                                                  \
            m_handle,                                                                                         \
            m_numberQubits,                                                                                   \
            m_adjoint,                                                                                        \
            targets,                                                                                          \
            m_stateVector,                                                                                    \
            m_extraWorkspace,                                                                                 \
            m_extraWorkspaceSizeInBytes                                                                       \
                ________SELECT_EXTRA_VARS(NUMBER_OF_EXTRA_PARAMS));                                           \
    }                                                                                                         \
                                                                                                              \
    template <precision selectedPrecision>                                                                    \
    void quantumState_SV<selectedPrecision>::GATE_NAME(std::initializer_list<const int> targets               \
                                                           ________SELECT_EXTRA_ARGS(NUMBER_OF_EXTRA_PARAMS), \
                                                       std::initializer_list<const int> controls)             \
    {                                                                                                         \
        apply##GATE_NAME<selectedPrecision>(                                                                  \
            m_handle,                                                                                         \
            m_numberQubits,                                                                                   \
            m_adjoint,                                                                                        \
            targets,                                                                                          \
            m_stateVector,                                                                                    \
            m_extraWorkspace,                                                                                 \
            m_extraWorkspaceSizeInBytes                                                                       \
                ________SELECT_EXTRA_VARS(NUMBER_OF_EXTRA_PARAMS));                                           \
    }
MAKE_GATES_BACKEND(X, 0)
MAKE_GATES_BACKEND(Y, 0)
MAKE_GATES_BACKEND(Z, 0)
MAKE_GATES_BACKEND(H, 0)

MAKE_GATES_BACKEND(RX, 1)
MAKE_GATES_BACKEND(RY, 1)
MAKE_GATES_BACKEND(RZ, 1)