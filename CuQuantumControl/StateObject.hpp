#pragma once
#include <custatevec.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <iostream>
#include <span>
#include <vector>
#include <cassert>

#include "ApplyGates.hpp"
#include "MatriceDefinitions.hpp"
#include "Precision.hpp"

template <precision selectedPrecision>
class quantumState_SV
{
private:
    using complex_t = PRECISION_TYPE_COMPLEX(selectedPrecision);

    complex_t *stateVector = nullptr;
    size_t stateVectorSize = 0;

    custatevecHandle_t handle;
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;
public:
    quantumState_SV();
    ~quantumState_SV();
    
};

template class quantumState_SV<precision::bit_32>;
template class quantumState_SV<precision::bit_64>;
