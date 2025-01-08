#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <bitset>   //Bitset
#include <cstring>  //memcpy

#define HANDLE_ERROR(x)                                                        \
{   const auto err = x;                                                        \
    if (err != CUSTATEVEC_STATUS_SUCCESS ) {                                   \
        printf("Error: %s in line %d\n",                                       \
               custatevecGetErrorString(err), __LINE__); return err; }         \
};

#define HANDLE_CUDA_ERROR(x)                                                   \
{   const auto err = x;                                                        \
    if (err != cudaSuccess ) {                                                 \
        printf("Error: %s in line %d\n",                                       \
               cudaGetErrorString(err), __LINE__); return err; }               \
};

int bugRunner()
{

    const int nIndexBits = 3;
    const int nSvSize = (1 << nIndexBits);
    const int nTargets = 1;
    const int nControls = 0;
    const int adjoint = 0;

    const int targets[] = {2};
    const int controls[] = {};

    //Pauli x
    cuDoubleComplex matrix[] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
    cuDoubleComplex h_sv[] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    cuDoubleComplex *d_sv;

    //Malloc device state vector
    HANDLE_CUDA_ERROR(cudaMallocManaged((void **)&d_sv, nSvSize * sizeof(cuDoubleComplex)));

    // Unified memory allows access directly to statevector from cpu
    std::memcpy(d_sv, h_sv, nSvSize * sizeof(cuDoubleComplex));


    //Workspace setup
    custatevecHandle_t handle;
    HANDLE_ERROR(custatevecCreate(&handle));
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    
    // check the size of external workspace
    HANDLE_ERROR(custatevecApplyMatrixGetWorkspaceSize(
        handle, CUDA_C_64F, nIndexBits, matrix, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
        adjoint, nTargets, nControls, CUSTATEVEC_COMPUTE_64F, &extraWorkspaceSizeInBytes));

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
        HANDLE_CUDA_ERROR(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));

    // apply gate
    HANDLE_ERROR(custatevecApplyMatrix(
        handle, d_sv, CUDA_C_64F, nIndexBits, matrix, CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets, nTargets, controls, nullptr,
        nControls, CUSTATEVEC_COMPUTE_64F, extraWorkspace, extraWorkspaceSizeInBytes));

    // The cause of the bug ------------------
    // If we forget to destroy handle
    // The bug will occur
    // HANDLE_ERROR(custatevecDestroy(handle));
    // The cause of the bug ------------------

    //Output
    //Expect 5,6,7,8,1,2,3,4
    //Bugged output 1,6,7,8,1,2,3,4
    for (int i = 0; i < nSvSize; i++)
    {
        std::cout << (d_sv[i].x) << "," << d_sv[i].y << " , " << static_cast<std::bitset<3>>(i) << std::endl;
    }
    std::cout << "\n";

    HANDLE_CUDA_ERROR(cudaFree(d_sv));

    return cudaSuccess;
}
