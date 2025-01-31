#pragma once
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
#include "../functionality/Utilities.hpp"
#include "ApplyGates.hpp"
#include "MatriceDefinitions.hpp"
#include "Precision.hpp"

template <precision selectedPrecision>
class quantumState_SV
{
private:
    using complex_type = PRECISION_TYPE_COMPLEX(selectedPrecision);
    
    complex_type *m_stateVector = nullptr;
    size_t m_numberQubits = 0;
    int m_adjoint = static_cast<int>(false);

    custatevecHandle_t m_handle;
    void *m_extraWorkspace = nullptr;
    size_t m_extraWorkspaceSizeInBytes = 0;

protected:
    void applyArbitaryGateUnsafe(std::span<const int> targets,
                                 std::span<const int> controls,
                                 std::span<const complex_type> matrix);

public:
    quantumState_SV(size_t nQubits);
    quantumState_SV(std::span<const complex_type> stateVector);
    ~quantumState_SV();

    std::span<complex_type> getStateVector();

    void setStateVector(std::span<const complex_type> stateVector);
    void setNumberOfQubits(size_t nQubits);

    void freeWorkspace();
    void freeStateVector();
    void prefetchToDevice(int deviceNumber = 0);
    void prefetchToCPU();

    void applyArbitaryGate(std::span<const int> targets,
                           std::span<const int> controls,
                           std::span<const complex_type> matrix);
    void applyArbitaryGate(std::initializer_list<const int> targets,
                           std::initializer_list<const int> controls,
                           std::initializer_list<const complex_type> matrix);
    // clang-format off
#define MAKE_GATES(GATE_NAME, NUMBER_OF_EXTRA_PARAMS)                     \
    void GATE_NAME(________SELECT_EXTRA_ARGS_PRE(NUMBER_OF_EXTRA_PARAMS)  \
                    std::span<const int> targets,                         \
                   std::span<const int> controls = {});                   \
                                                                          \
    void GATE_NAME(________SELECT_EXTRA_ARGS_PRE(NUMBER_OF_EXTRA_PARAMS)  \
                   std::initializer_list<const int> targets,              \
                   std::initializer_list<const int> controls = {});
    // clang-format on


    MAKE_GATES(X, 0)
    MAKE_GATES(Y, 0)
    MAKE_GATES(Z, 0)
    MAKE_GATES(H, 0)

    MAKE_GATES(RX, 1)
    MAKE_GATES(RY, 1)
    MAKE_GATES(RZ, 1)
};

template class quantumState_SV<precision::bit_32>;
template class quantumState_SV<precision::bit_64>;
