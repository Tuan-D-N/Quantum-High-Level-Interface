#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <bitset>
#include "helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR

int ApplyQFTOnStateVector(cuDoubleComplex *d_stateVector, int numQubits)
{
    const int nSvSize = (1 << numQubits);
}