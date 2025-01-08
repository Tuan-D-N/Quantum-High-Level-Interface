#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <bitset>
#include "helper.hpp" // HANDLE_ERROR, HANDLE_CUDA_ERROR
#include "ApplyGates.hpp"
#include <cstring>
#include "QftRhoWise.hpp"

int ApplyQFTOnStateVector(cuDoubleComplex *d_stateVector, int numQubits)
{
    const int nSvSize = (1 << numQubits);
    const int adjoint = static_cast<int>(false);
    const int nTargets = 1;

    custatevecHandle_t handle;
    HANDLE_ERROR(custatevecCreate(&handle));
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    for (int i_qubit = 0; i_qubit < numQubits; ++i_qubit)
    {
        int i_qubit_reversed = numQubits - 1 - i_qubit;
        const int targets[] = {i_qubit_reversed};
        applyH(handle, numQubits, adjoint, i_qubit_reversed, d_stateVector, extraWorkspace, extraWorkspaceSizeInBytes);

        for(int j_qubit = 0; j_qubit < i_qubit_reversed; ++j_qubit )
        {
            int n = j_qubit + 2;
            const int controls[] = {i_qubit - 1 - j_qubit};
            const int ncontrols = 1;
            applyGatesGeneral(handle, numQubits, {}, adjoint, targets, nTargets, controls, ncontrols, d_stateVector, extraWorkspace, extraWorkspaceSizeInBytes);
            
        }

    }

    // destroy handle
    HANDLE_ERROR(custatevecDestroy(handle));

    return cudaSuccess;
}