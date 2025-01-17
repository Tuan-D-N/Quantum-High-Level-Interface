#pragma once
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include "Precision.hpp"

template <precision SelectPrecision = precision::bit_64>
int ApplyQFTOnStateVector(PRECISION_TYPE_COMPLEX(SelectPrecision) *d_stateVector, int numQubits);