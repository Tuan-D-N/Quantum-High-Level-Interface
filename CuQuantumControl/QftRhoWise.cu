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
#include "SwapGates.hpp"
#include <vector>

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

        HANDLE_ERROR(
            applyH(handle, numQubits, adjoint, i_qubit_reversed, d_stateVector, extraWorkspace, extraWorkspaceSizeInBytes));

        for (int j_qubit = 0; j_qubit < i_qubit_reversed; ++j_qubit)
        {
            int n = j_qubit + 2;
            const int controls[] = {i_qubit_reversed - 1 - j_qubit};
            const int ncontrols = 1;
            const cuDoubleComplex matrix[] = RKMat(n);

            // std::cout << "num Qubits" << numQubits << "\n";
            // for (int i = 0; i < 4; ++i)
            // {
            //     std::cout << matrix[i].x << "," << matrix[i].y << "\n";
            // }
            // std::cout << "adjoint" << adjoint << "\n";
            // std::cout << "Targets" << targets[0] << "\n";
            // std::cout << "nTargets" << nTargets << "\n";
            // std::cout << "controls" << controls[0] << "\n";
            // std::cout << "ncontrols" << ncontrols << "\n";
            // for (int i = 0; i < nSvSize; ++i)
            // {
            //     std::cout << d_stateVector[i].x << "," << d_stateVector[i].y << "\n";
            // }
            // std::cout << "extraWorkspaceSizeInBytes" << extraWorkspaceSizeInBytes << "\n";

            HANDLE_ERROR(
                static_cast<custatevecStatus_t>(applyGatesGeneral(handle,
                                                                  numQubits,
                                                                  matrix,
                                                                  adjoint,
                                                                  targets,
                                                                  nTargets,
                                                                  controls,
                                                                  ncontrols,
                                                                  d_stateVector,
                                                                  extraWorkspace,
                                                                  extraWorkspaceSizeInBytes)));
        }
    }

    const int numberOfSwaps = numQubits / 2;
    std::vector<int2> qubitsToSwap;
    qubitsToSwap.reserve(numberOfSwaps);
    for (int i = 0; i < numberOfSwaps; ++i)
    {
        int swapA = i;
        int swapB = numQubits - i - 1;
        qubitsToSwap[i] = {swapA, swapB};
    }
    swap(handle, numQubits, qubitsToSwap.data(), numberOfSwaps, d_stateVector);
    // destroy handle
    HANDLE_ERROR(custatevecDestroy(handle));

    return cudaSuccess;
}