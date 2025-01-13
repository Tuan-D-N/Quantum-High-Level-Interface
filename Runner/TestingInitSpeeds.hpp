#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecInitializeStateVector
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "../CudaControl/Helper.hpp" // CHECK_CUSTATEVECTOR, CHECK_CUDA
template<int nIndexBits>
int initByFunction()
{
    const int svSize = (1 << nIndexBits);

    cuDoubleComplex h_sv[svSize];

    cuDoubleComplex *d_sv;
    CHECK_CUDA(cudaMalloc((void **)&d_sv, svSize * sizeof(cuDoubleComplex)));

    // populate the device memory with junk values (for illustrative purpose only)
    CHECK_CUDA(cudaMemset(d_sv, 0x7F, svSize * sizeof(cuDoubleComplex)));

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    CHECK_CUSTATEVECTOR(custatevecCreate(&handle));

    // initialize the state vector
    CHECK_CUSTATEVECTOR(custatevecInitializeStateVector(
        handle, d_sv, CUDA_C_64F, nIndexBits, CUSTATEVEC_STATE_VECTOR_TYPE_ZERO));

    // destroy handle
    CHECK_CUSTATEVECTOR(custatevecDestroy(handle));

    //----------------------------------------------------------------------------------------------

    CHECK_CUDA(cudaMemcpy(h_sv, d_sv, svSize * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_sv));

    return EXIT_SUCCESS;
}

template<int nIndexBits>
int initByHand() //Significantly faster
{
    const int svSize = (1 << nIndexBits);

    cuDoubleComplex h_sv[svSize];

    cuDoubleComplex *d_sv;
    CHECK_CUDA(cudaMallocManaged((void **)&d_sv, svSize * sizeof(cuDoubleComplex)));

    // populate the device memory with junk values (for illustrative purpose only)
    CHECK_CUDA(cudaMemset(d_sv, 0x7F, svSize * sizeof(cuDoubleComplex)));

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    CHECK_CUSTATEVECTOR(custatevecCreate(&handle));

    
    d_sv[0] = {1, 0};
    for (int i = 1; i < svSize; ++i)
    {
        d_sv[i] = {0, 0};
    }

    // destroy handle
    CHECK_CUSTATEVECTOR(custatevecDestroy(handle));

    //----------------------------------------------------------------------------------------------

    CHECK_CUDA(cudaMemcpy(h_sv, d_sv, svSize * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_sv));

    return EXIT_SUCCESS;
}