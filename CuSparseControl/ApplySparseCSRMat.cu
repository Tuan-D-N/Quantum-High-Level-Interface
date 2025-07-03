#include "ApplySparseCSRMat.hpp"

int applySparseCSRMat(cusparseHandle_t handle,
                      std::span<int> csrOffsets,
                      std::span<int> csrRows,
                      std::span<cuDoubleComplex> values,
                      std::span<cuDoubleComplex> svInput,
                      std::span<cuDoubleComplex> svOutput)
{
    cusparseSpMatDescr_t matrixOBJ;
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    int matrix_num_rows = svInput.size();
    int matrix_num_cols = svInput.size();

    CHECK_CUSPARSE(cusparseCreateCsr(&matrixOBJ, matrix_num_rows, matrix_num_cols, values.size(),
                                     csrOffsets.data(), csrRows.data(), values.data(),
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F))

    cusparseDnVecDescr_t vectorIn;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vectorIn, svInput.size(), svInput.data(), CUDA_C_64F));

    cusparseDnVecDescr_t vectorOut;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vectorOut, svOutput.size(), svOutput.data(), CUDA_C_64F));

    //---------------------------------------------------------------------------

    // Workspace buffer
    void *dBuffer = nullptr;

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE,
                                           &alpha, matrixOBJ, vectorIn, &beta, vectorOut,
                                           CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // Perform the SpMV operation
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE,
                                &alpha, matrixOBJ, vectorIn, &beta, vectorOut,
                                CUDA_C_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    cusparseDestroySpMat(matrixOBJ);
    cusparseDestroyDnVec(vectorIn);
    cusparseDestroyDnVec(vectorOut);
    
    return cudaSuccess;
}