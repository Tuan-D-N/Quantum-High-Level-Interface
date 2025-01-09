
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include "../functionality/WriteAdjMat.hpp"
#include "../functionality/ReadCsv.hpp"
#include "../functionality/Utilities.hpp"
#include "../functionality/fftShift.hpp"
#include "../CuQuantumControl/QftStateVec.hpp"
#include <cuComplex.h>
#include <iostream>
#include "Sys.hpp"
#include <string>
#include <cassert>

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

template <typename T>
void printDeviceArray(T *d_array, T size)
{
    T *h_array = new T[size];
    cudaMemcpy(h_array, d_array, size * sizeof(T), cudaMemcpyDeviceToHost);
    for (T i = 0; i < size; ++i)
        std::cout << h_array[i] << " ";
    std::cout << std::endl;
    delete[] h_array;
}

void printDeviceArray(cuDoubleComplex *d_array, int size)
{
    cuDoubleComplex *h_array = new cuDoubleComplex[size];
    cudaMemcpy(h_array, d_array, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i)
        std::cout << "(" << h_array[i].x << ", " << h_array[i].y << ") ";
    std::cout << std::endl;

    delete[] h_array;
}

void getData(cuDoubleComplex *rThetaVector, const int evenqubits, const std::string fileName)
{
    int lengthSize = 1 << (evenqubits / 2);
    std::vector<std::vector<float>> image = readCSV<float>(fileName);
    assert(image.size() == lengthSize);

    for (int i = 0; i < lengthSize; ++i)
    {
        assert(image[i].size() == lengthSize);
        for (int j = 0; j < lengthSize; ++j)
        {
            rThetaVector[i * lengthSize + j] = {image[i][j], 0};
        }
    }
}

void applyQFTHorizontally(cuDoubleComplex *vector, const int num_columns, const int num_rows, const int num_qubit_per_row)
{
    for (int i = 0; i < num_rows; ++i)
    {
        ApplyQFTOnStateVector(&vector[i * num_columns], num_qubit_per_row);
    }
}

void applyQFTVertically(cuDoubleComplex *vector, cuDoubleComplex *workSpace, const int num_columns, const int num_rows, const int num_qubit_per_row)
{
    for (int i = 0; i < num_columns; ++i)
    {
        for (int j = 0; j < num_rows; ++j)
        {
            workSpace[j] = vector[i * num_rows + j];
        }

        ApplyQFTOnStateVector(workSpace, num_qubit_per_row);

        for (int j = 0; j < num_rows; ++j)
        {
            vector[i * num_rows + j] = workSpace[j];
        }
    }
}

int runSys()
{
    // Host problem definition
    int evenqubits = 4;
    int halfOfQubits = evenqubits / 2;
    int svSize = 1 << evenqubits;
    int img_num_rows = 1 << (halfOfQubits);
    int img_num_columns = 1 << (halfOfQubits);
    int A_num_rows = 1 << evenqubits;
    int A_num_cols = 1 << evenqubits;
    int A_max_nnz = 4 * A_num_rows;

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    //--------------------------------------------------------------------------
    // Device memory management
    int *dA_csrOffsets, *dA_columns;
    cuDoubleComplex *dA_values;
    CHECK_CUDA(cudaMallocManaged((void **)&dA_csrOffsets,
                                 (A_num_rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMallocManaged((void **)&dA_columns, A_max_nnz * sizeof(int)))
    CHECK_CUDA(cudaMallocManaged((void **)&dA_values, A_max_nnz * sizeof(cuDoubleComplex)))

    int postIndexSize, postOffsetSize, postValueSize;

    // Unified Memory Cuda Write
    writeMatAMiniCSC(dA_csrOffsets, dA_columns, dA_values, evenqubits, postOffsetSize, postIndexSize, postValueSize);

    // Vector
    cuDoubleComplex *rThetaVector; // theta slow, r fast
    cuDoubleComplex *xyVector;
    CHECK_CUDA(cudaMallocManaged((void **)&rThetaVector, A_num_cols * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMallocManaged((void **)&xyVector, A_num_cols * sizeof(cuDoubleComplex)));

    for (int i = 0; i < A_num_cols; ++i)
    {
        xyVector[0] = make_cuDoubleComplex(0, 0);
        rThetaVector[i] = make_cuDoubleComplex(0, 0);
    }
    getData(rThetaVector, evenqubits, "../imageFile.csv");

    printDeviceArray(rThetaVector, svSize);
    fftshift2D(rThetaVector, img_num_rows, img_num_columns);
    printDeviceArray(rThetaVector, svSize);

    cuDoubleComplex *qftWorkSpace;
    CHECK_CUDA(cudaMallocManaged((void **)&qftWorkSpace, img_num_rows * sizeof(cuDoubleComplex)));

    for (int i = 0; i < img_num_rows; ++i)
    {
        ApplyQFTOnStateVector(&rThetaVector[i * img_num_columns], halfOfQubits);
    }
    printDeviceArray(rThetaVector, svSize);
    
    CHECK_CUDA(cudaFree(qftWorkSpace))
    //--------------------------------------------------------------------------
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreate(&handle))

    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, postValueSize,
                                     dA_csrOffsets, dA_columns, dA_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F))

    cusparseDnVecDescr_t vectorIn;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vectorIn, A_num_rows, rThetaVector, CUDA_C_64F));

    cusparseDnVecDescr_t vectorOut;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vectorOut, A_num_rows, xyVector, CUDA_C_64F));

    //---------------------------------------------------------------------------

    // Workspace buffer
    void *dBuffer = nullptr;
    size_t bufferSize = 0;
    float tmp_result;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE,
                                           &alpha, matA, vectorIn, &beta, vectorOut,
                                           CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // Perform the SpMV operation
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE,
                                &alpha, matA, vectorIn, &beta, vectorOut,
                                CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    printDeviceArray(dA_csrOffsets, postOffsetSize);
    printDeviceArray(dA_columns, postIndexSize);
    printDeviceArray(dA_values, postValueSize);

    std::cout << alpha.x << "," << alpha.y << "\n";
    std::cout << beta.x << "," << beta.y << "\n";

    printDeviceArray(rThetaVector, A_num_cols);
    printDeviceArray(xyVector, A_num_cols);

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vectorIn);
    cusparseDestroyDnVec(vectorOut);


    CHECK_CUDA(cudaFree(dA_csrOffsets))
    CHECK_CUDA(cudaFree(dA_columns))
    CHECK_CUDA(cudaFree(dA_values))
    CHECK_CUDA(cudaFree(xyVector))
    CHECK_CUDA(cudaFree(rThetaVector))
    return EXIT_SUCCESS;
}