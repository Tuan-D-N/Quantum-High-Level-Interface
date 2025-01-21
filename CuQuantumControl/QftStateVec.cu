#include <cuda_runtime_api.h> 
#include <cuComplex.h>        
#include <custatevec.h>       
#include <stdio.h>            
#include <stdlib.h>           
#include <iostream>
#include <bitset>
#include "../CudaControl/Helper.hpp" 
#include "ApplyGates.hpp"
#include <cstring>
#include "QftStateVec.hpp"
#include "SwapGates.hpp"
#include "Precision.hpp"
#include <vector>

template <precision SelectPrecision>
int ApplyQFTOnStateVector(PRECISION_TYPE_COMPLEX(SelectPrecision) *d_stateVector, int numQubits)
{
    const int adjoint = static_cast<int>(false);
    const int nTargets = 1;

    custatevecHandle_t handle;
    CHECK_CUSTATEVECTOR(custatevecCreate(&handle));
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    for (int i_qubit = 0; i_qubit < numQubits; ++i_qubit)
    {
        int i_qubit_reversed = numQubits - 1 - i_qubit;
        const int targets[] = {i_qubit_reversed};
        CHECK_BROAD_ERROR(applyH(handle, numQubits, adjoint, i_qubit_reversed, d_stateVector, extraWorkspace, extraWorkspaceSizeInBytes));

        //The controled rotation loop
        for (int j_qubit = 0; j_qubit < i_qubit_reversed; ++j_qubit)
        {
            int n = j_qubit + 2;
            const int controls[] = {i_qubit_reversed - 1 - j_qubit};
            const int ncontrols = 1;
            const PRECISION_TYPE_COMPLEX(SelectPrecision) matrix[] = RKMat(n);
            (
                (applyGatesGeneral<precision::bit_64>(handle,
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
    swap<SelectPrecision>(handle, numQubits, qubitsToSwap.data(), numberOfSwaps, d_stateVector);

    // destroy handle
    CHECK_CUSTATEVECTOR(custatevecDestroy(handle));

    return cudaSuccess;
}