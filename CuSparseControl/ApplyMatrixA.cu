#include "ApplyMatrixA.hpp"
#include <stdlib.h>           // EXIT_FAILURE
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>
#include <cusparse.h>
#include "../CudaControl/Helper.hpp"
#include "../functionality/WriteAdjMat.hpp"

// theta slow, r fast
int applyInterpolationMatrix(int evenqubits, cuDoubleComplex *rThetaVector, cuDoubleComplex *&xyVector)
{
    // Host problem definition
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
    writeMatAMiniCSC(dA_csrOffsets, dA_columns, dA_values, evenqubits, postOffsetSize, postIndexSize, postValueSize, false);

    // Vector
    if (xyVector == nullptr)
    {
        CHECK_CUDA(cudaMallocManaged((void **)&xyVector, A_num_cols * sizeof(cuDoubleComplex)));
        for (int i = 0; i < A_num_cols; ++i)
        {
            xyVector[i] = {0, 0};
        }
    }

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
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE,
                                           &alpha, matA, vectorIn, &beta, vectorOut,
                                           CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // Perform the SpMV operation
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE,
                                &alpha, matA, vectorIn, &beta, vectorOut,
                                CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vectorIn);
    cusparseDestroyDnVec(vectorOut);

    CHECK_CUDA(cudaFree(dA_csrOffsets))
    CHECK_CUDA(cudaFree(dA_columns))
    CHECK_CUDA(cudaFree(dA_values))

    return cudaSuccess;
}