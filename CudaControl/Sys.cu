
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include "../functionality/WriteAdjMat.hpp"
#include <cuComplex.h>
#include <iostream>
#include "Sys.hpp"

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }

template<typename T>
void printDeviceArray(T* d_array, T size) {
    T* h_array = new T[size];
    cudaMemcpy(h_array, d_array, size * sizeof(T), cudaMemcpyDeviceToHost);
    for (T i = 0; i < size; ++i)
        std::cout << h_array[i] << " ";
    std::cout << std::endl;
    delete[] h_array;
}

void printDeviceArray(cuDoubleComplex* d_array, int size) {
    cuDoubleComplex* h_array = new cuDoubleComplex[size];
    cudaMemcpy(h_array, d_array, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < size; ++i)
        std::cout << "(" << h_array[i].x << ", " << h_array[i].y << ") ";
    std::cout << std::endl;
    
    delete[] h_array;
}

int runSys()
{
    // Host problem definition
    int evenqubits = 4;
    int A_num_rows = 1 << evenqubits;
    int A_num_cols = 1 << evenqubits;
    int A_max_nnz = 4*A_num_rows;

    float alpha = 1.0f;
    float beta = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int *dA_csrOffsets, *dA_columns;
    cuDoubleComplex *dA_values;
    CHECK_CUDA(cudaMallocManaged((void **)&dA_csrOffsets,
                                 (A_num_rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMallocManaged((void **)&dA_columns, A_max_nnz * sizeof(int)))
    CHECK_CUDA(cudaMallocManaged((void **)&dA_values, A_max_nnz * sizeof(cuDoubleComplex)))

    int postIndexSize, postOffsetSize, postValueSize;

    writeMatAMiniCSC(dA_csrOffsets, dA_columns, dA_values, evenqubits, postOffsetSize, postIndexSize, postValueSize);
    //--------------------------------------------------------------------------

    printDeviceArray(dA_csrOffsets, postOffsetSize);
    printDeviceArray(dA_columns, postOffsetSize);
    printDeviceArray(dA_values, postOffsetSize);
    

    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    return EXIT_SUCCESS;
}