
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
#include <c++/11/bits/specfun.h>

#ifdef __CUDACC__
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#else
#define CUDA_KERNEL(...) 
#endif

#define nullify

#define CUDA_CHECK(err)                                                                                      \
    do                                                                                                       \
    {                                                                                                        \
        cudaError_t err_ = (err);                                                                            \
        if (err_ != cudaSuccess)                                                                             \
        {                                                                                                    \
            std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(err_) << std::endl; \
            std::exit(EXIT_FAILURE);                                                                         \
        }                                                                                                    \
    } while (0)

#define CUSPARSE_CHECK(stat)                                                 \
    do                                                                       \
    {                                                                        \
        cusparseStatus_t stat_ = (stat);                                     \
        if (stat_ != CUSPARSE_STATUS_SUCCESS)                                \
        {                                                                    \
            std::cerr << "cuSPARSE error at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

int calcGridSize(int64_t N, int blockSize)
{
    return (N + blockSize - 1) / blockSize;
}

double norm_squared(const std::vector<cuDoubleComplex> &v)
{
    double sum = 0.0;
    for (auto &elem : v)
    {
        double re = cuCreal(elem);
        double im = cuCimag(elem);
        sum += re * re + im * im;
    }
    return sum;
}

// Hypercube adjacency matrix in the CSR format - only generating rowOffsets and colIndices.
// For the values we fill an array of size numVertices with ones.
void generateHypercubeCSR(int n, std::vector<int64_t> &rowOffsets, std::vector<int64_t> &colIndices)
{
    int64_t numVertices = (int64_t)1 << n;

    rowOffsets.resize((size_t)numVertices + 1);
    colIndices.reserve((size_t)n * (size_t)numVertices); // each vertex has n neighbors

    int64_t edgeCount = 0;
    for (int64_t i = 0; i < numVertices; ++i)
    {
        rowOffsets[(size_t)i] = edgeCount;
        for (int bit = 0; bit < n; ++bit)
        {
            int64_t neighbor = i ^ ((int64_t)1 << bit);
            colIndices.push_back(neighbor);
            edgeCount++;
        }
    }
    rowOffsets[(size_t)numVertices] = edgeCount;
}

__global__ void scale_kernel(cuDoubleComplex *x, int64_t N, double alpha_real, double alpha_imag)
{
    cuDoubleComplex alpha = make_cuDoubleComplex(alpha_real, alpha_imag);
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        x[idx] = cuCmul(alpha, x[idx]);
    }
}

__global__ void axpy_kernel(cuDoubleComplex *y, const cuDoubleComplex *x, int64_t N, double alpha_real, double alpha_imag)
{
    cuDoubleComplex alpha = make_cuDoubleComplex(alpha_real, alpha_imag);
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        y[idx] = cuCadd(y[idx], cuCmul(alpha, x[idx]));
    }
}

void scale_vector(cuDoubleComplex *x, int64_t N, std::complex<double> alpha)
{
    int blockSize = 256;
    int64_t gridSize = calcGridSize(N, blockSize);
    scale_kernel CUDA_KERNEL((unsigned int)gridSize, blockSize) (x, N, alpha.real(), alpha.imag());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void axpy_vector(cuDoubleComplex *y, const cuDoubleComplex *x, int64_t N, std::complex<double> alpha)
{
    int blockSize = 256;
    int64_t gridSize = calcGridSize(N, blockSize);
    axpy_kernel CUDA_KERNEL((unsigned int)gridSize, blockSize) (y, x, N, alpha.real(), alpha.imag());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main()
{
    int n = 6; // number of qubits
    int64_t numVertices = (int64_t)1 << n;

    std::vector<int64_t> rowOffsets, colIndices; // using 64-bit indexing for n > 31
    generateHypercubeCSR(n, rowOffsets, colIndices);
    int64_t nnz = (int64_t)colIndices.size();

    double M = double(n);
    double t = 1.0;

    // CSR values array (all ones)
    std::vector<cuDoubleComplex> h_csrVal((size_t)nnz);
    std::fill(h_csrVal.begin(), h_csrVal.end(), make_cuDoubleComplex(1.0, 0.0));

    // Initial state
    std::vector<cuDoubleComplex> h_psi((size_t)numVertices);
    std::fill(h_psi.begin(), h_psi.end(), make_cuDoubleComplex(0.0, 0.0));
    h_psi[0].x = 1.0;

    // Allocate and transfer to device memory
    cuDoubleComplex *d_csrVal, *d_psi, *d_w0, *d_w1, *d_result;
    int64_t *d_csrRowPtr, *d_csrColInd;

    CUDA_CHECK(cudaMalloc((void **)&d_csrVal, sizeof(cuDoubleComplex) * nnz));
    CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtr, sizeof(int64_t) * (numVertices + 1)));
    CUDA_CHECK(cudaMalloc((void **)&d_csrColInd, sizeof(int64_t) * nnz));

    CUDA_CHECK(cudaMemcpy(d_csrVal, h_csrVal.data(), sizeof(cuDoubleComplex) * nnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrRowPtr, rowOffsets.data(), sizeof(int64_t) * (numVertices + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrColInd, colIndices.data(), sizeof(int64_t) * nnz, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_psi, sizeof(cuDoubleComplex) * numVertices));
    CUDA_CHECK(cudaMalloc((void **)&d_w0, sizeof(cuDoubleComplex) * numVertices));
    CUDA_CHECK(cudaMalloc((void **)&d_w1, sizeof(cuDoubleComplex) * numVertices));
    CUDA_CHECK(cudaMalloc((void **)&d_result, sizeof(cuDoubleComplex) * numVertices));

    CUDA_CHECK(cudaMemcpy(d_psi, h_psi.data(), sizeof(cuDoubleComplex) * numVertices, cudaMemcpyHostToDevice));

    // Create cuSPARSE handle and descriptors
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    cusparseSpMatDescr_t matDescr;
    cusparseDnVecDescr_t vecX, vecY;

    // Using 64-bit indexing in cuSPARSE
    CUSPARSE_CHECK(cusparseCreateCsr(&matDescr,
                                     (int64_t)numVertices, (int64_t)numVertices, nnz,
                                     d_csrRowPtr, d_csrColInd, d_csrVal,
                                     CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));

    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, (int64_t)numVertices, d_w0, CUDA_C_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, (int64_t)numVertices, d_w1, CUDA_C_64F));

    // Buffer for SpMV
    size_t bufferSize = 0;
    void *dBuffer = nullptr;
    cuDoubleComplex alpha = {1.0, 0.0}, beta = {0.0, 0.0};
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           matDescr,
                                           vecX,
                                           &beta,
                                           vecY,
                                           CUDA_C_64F,
                                           CUSPARSE_SPMV_ALG_DEFAULT,
                                           &bufferSize));
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    double epsilon = 1e-15; // target precision
    double z = t * M;
    std::complex<double> minus_i(0.0, -1.0);

    // Determine the polynomial degree m
    std::vector<std::complex<double>> ck;
    int k = 0;

    double J0 = std::cyl_bessel_j(0, z);
    ck.push_back(J0); // c0 = J0(z)

    if (std::abs(J0) >= epsilon)
    {
        while (true)
        {
            k++;
            double Jk = std::cyl_bessel_j(k, z);
            std::complex<double> ck_val = 2.0 * std::pow(minus_i, k) * Jk;
            ck.push_back(ck_val);

            if (std::abs(Jk) < epsilon)
            {
                break;
            }
        }
    }

    int m = k;
    std::cout << "Selected polynomial degree m = " << m << std::endl;

    // w0 = v
    CUDA_CHECK(cudaMemcpy(d_w0, d_psi, sizeof(cuDoubleComplex) * numVertices, cudaMemcpyDeviceToDevice));

    // Scale to get H_tilde = H/M
    for (int64_t i = 0; i < nnz; i++)
    {
        h_csrVal[(size_t)i] = make_cuDoubleComplex(1.0 / M, 0.0);
    }
    CUDA_CHECK(cudaMemcpy(d_csrVal, h_csrVal.data(), sizeof(cuDoubleComplex) * nnz, cudaMemcpyHostToDevice));

    // w1 = H_tilde * w0
    CUSPARSE_CHECK(cusparseDnVecSetValues(vecX, d_w0));
    CUSPARSE_CHECK(cusparseDnVecSetValues(vecY, d_w1));
    cusparseSpMV(handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha,
                 matDescr,
                 vecX,
                 &beta,
                 vecY,
                 CUDA_C_64F,
                 CUSPARSE_SPMV_ALG_DEFAULT,
                 dBuffer);

    // result = c0*w0 + c1*w1
    CUDA_CHECK(cudaMemcpy(d_result, d_w0, sizeof(cuDoubleComplex) * numVertices, cudaMemcpyDeviceToDevice));
    scale_vector(d_result, numVertices, ck[0]);
    axpy_vector(d_result, d_w1, numVertices, ck[1]);

    cuDoubleComplex *d_wtemp;
    CUDA_CHECK(cudaMalloc((void **)&d_wtemp, sizeof(cuDoubleComplex) * numVertices));

    for (int i_iter = 2; i_iter <= m; i_iter++)
    {
        CUDA_CHECK(cudaMemcpy(d_wtemp, d_w0, sizeof(cuDoubleComplex) * numVertices, cudaMemcpyDeviceToDevice));

        // w0 = w1
        CUDA_CHECK(cudaMemcpy(d_w0, d_w1, sizeof(cuDoubleComplex) * numVertices, cudaMemcpyDeviceToDevice));

        // w1 = H_tilde*w0
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecX, d_w0));
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecY, d_w1));

        cusparseSpMV(handle,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha,
                     matDescr,
                     vecX,
                     &beta,
                     vecY,
                     CUDA_C_64F,
                     CUSPARSE_SPMV_ALG_DEFAULT,
                     dBuffer);

        // w1 = 2*w1 - wtemp
        scale_vector(d_w1, numVertices, std::complex<double>(2.0, 0.0));
        axpy_vector(d_w1, d_wtemp, numVertices, std::complex<double>(-1.0, 0.0));

        // result += c_i*w1
        axpy_vector(d_result, d_w1, numVertices, ck[i_iter]);
    }

    CUDA_CHECK(cudaFree(d_wtemp));

    // Copy result back to host - can avoid this by using managed memory...
    std::vector<cuDoubleComplex> h_result((size_t)numVertices);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, sizeof(cuDoubleComplex) * numVertices, cudaMemcpyDeviceToHost));

    std::cout << "Result (first few elements):\n";
    for (int64_t i = 0; i < std::min<int64_t>(numVertices, 5); i++)
    {
        std::complex<double> val(cuCreal(h_result[(size_t)i]), cuCimag(h_result[(size_t)i]));
        std::cout << val << "\n";
    }

    double norm = norm_squared(h_result);
    std::cout << norm << std::endl;

    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    CUSPARSE_CHECK(cusparseDestroySpMat(matDescr));
    CUSPARSE_CHECK(cusparseDestroy(handle));

    CUDA_CHECK(cudaFree(d_csrVal));
    CUDA_CHECK(cudaFree(d_csrRowPtr));
    CUDA_CHECK(cudaFree(d_csrColInd));
    CUDA_CHECK(cudaFree(d_psi));
    CUDA_CHECK(cudaFree(d_w0));
    CUDA_CHECK(cudaFree(d_w1));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(dBuffer));

    return 0;
}