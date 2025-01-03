#include <complex>
#include <vector>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <cstdlib>
#include <cstdint>
#include <device_launch_parameters.h>
#include <iostream>

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(err_) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int main(void) {
    CUDA_CHECK(cudaMalloc((void**) 1, 1 * sizeof(float)));
    // Host problem definition
    int   num_rows     = 4;
    int   num_cols     = 6;
    int   ld           = num_cols;
    int   dense_size   = ld * num_rows;
    float h_dense[]    = {0.0f,  0.0f,  1.0f,  2.0f,  0.0f,  0.0f,
                          0.0f,  0.0f,  3.0f,  4.0f,  0.0f,  0.0f,
                          5.0f,  6.0f,  0.0f,  0.0f,  7.0f,  8.0f,
                          9.0f, 10.0f,  0.0f,  0.0f, 11.0f, 12.0f };
    int   ell_blk_size = 2;
    int   ell_width    = 4;
    int   nnz          = ell_width * num_rows;
    int   h_ell_columns[]         = {1, -1, 0, 2};
    float h_ell_values[]          = {0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0};
    float h_ell_values_result[]   = {1.0f,  2.0f,  0.0f,  0.0f,
                                     3.0f,  4.0f,  0.0f,  0.0f,
                                     5.0f,  6.0f,  7.0f,  8.0f,
                                     9.0f, 10.0f, 11.0f, 12.0f};
    //--------------------------------------------------------------------------
    // Device memory management
    int   *d_ell_columns;
    float *d_ell_values,  *d_dense;
    CHECK_CUDA( cudaMalloc((void**) &d_dense, dense_size * sizeof(float)))
    CHECK_CUDA( cudaMalloc((void**) &d_ell_columns,
                           nnz / (ell_blk_size * ell_blk_size) * sizeof(int)))
    CHECK_CUDA( cudaMalloc((void**) &d_ell_values,
                           nnz * sizeof(float)))
    CHECK_CUDA( cudaMemcpy(d_dense, h_dense, dense_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_ell_columns, h_ell_columns,
                           nnz / (ell_blk_size * ell_blk_size) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_ell_values, h_ell_values,
                           nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, num_rows, num_cols, ld, d_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )

    // Create sparse matrix B in Blocked ELL format
    CHECK_CUSPARSE( cusparseCreateBlockedEll(&matB, num_rows, num_cols,
                                             ell_blk_size, ell_width,
                                             d_ell_columns, d_ell_values,
                                             CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO,
                                             CUDA_R_32F) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
                                        handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )

    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(h_ell_values, d_ell_values,
                           nnz * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if (h_ell_values[i] != h_ell_values_result[i]) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("dense2sparse_blockedell_example test PASSED\n");
    else
        printf("dense2sparse_blockedell_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(d_ell_columns) )
    CHECK_CUDA( cudaFree(d_ell_values) )
    CHECK_CUDA( cudaFree(d_dense) )
    return EXIT_SUCCESS;
}