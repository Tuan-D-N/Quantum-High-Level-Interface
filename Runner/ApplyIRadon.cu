
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include "../functionality/WriteAdjMat.hpp"
#include "../functionality/ReadCsv.hpp"
#include "../functionality/Utilities.hpp"
#include "../functionality/fftShift.hpp"
#include "../functionality/Transpose.hpp"
#include "../CuQuantumControl/QftStateVec.hpp"
#include "../CudaControl/Helper.hpp"
#include "../CuSparseControl/ApplyMatrixA.hpp"
#include <cuComplex.h>
#include <iostream>
#include "ApplyIRadon.hpp"
#include <string>
#include <cassert>

void getData(cuDoubleComplex *rThetaVector, const int evenqubits, const std::string fileName)
{
    int lengthSize = 1 << (evenqubits / 2);
    std::vector<std::vector<float>> image = readCSV<float>(fileName);
    if (image.size() != lengthSize)
    {
        std::cout << "imageSize: " << image.size() << "\n";
        std::cout << "lengthSize: " << lengthSize << "\n";
        assert(image.size() == lengthSize);
    }

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
            workSpace[j] = vector[j * num_rows + i];
        }
        ApplyQFTOnStateVector(workSpace, num_qubit_per_row);

        for (int j = 0; j < num_rows; ++j)
        {
            vector[j * num_rows + i] = workSpace[j];
        }
    }
}

int runSys()
{
    // Host problem definition
    int evenqubits = 12;
    int halfOfQubits = evenqubits / 2;
    int img_num_rows = 1 << (halfOfQubits);
    int img_num_columns = 1 << (halfOfQubits);
    int A_num_cols = 1 << evenqubits;

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    //--------------------------------------------------------------------------
    // Vector
    cuDoubleComplex *rThetaVector; // theta slow, r fast
    cuDoubleComplex *xyVector = nullptr;
    CHECK_CUDA(cudaMallocManaged((void **)&rThetaVector, A_num_cols * sizeof(cuDoubleComplex)));

    cuDoubleComplex *qftWorkSpace;
    CHECK_CUDA(cudaMallocManaged((void **)&qftWorkSpace, img_num_rows * sizeof(cuDoubleComplex)));

    getData(rThetaVector, evenqubits, "../imageFile.csv");

    printDeviceArray(rThetaVector, A_num_cols);
    fftshift2D(rThetaVector, img_num_rows, img_num_columns);
    applyQFTVertically(rThetaVector, qftWorkSpace, img_num_columns, img_num_rows, halfOfQubits);
    fftshift2D(rThetaVector, img_num_rows, img_num_columns);
    printDeviceArray(rThetaVector, A_num_cols);

    Transpose(rThetaVector, img_num_rows, img_num_columns);
    CHECK_CUDA(static_cast<cudaError_t>(applyInterpolationMatrix(evenqubits, rThetaVector, xyVector)));
    printDeviceArray(xyVector, A_num_cols);

    fftshift2D(xyVector, img_num_rows, img_num_columns);
    applyQFTHorizontally(xyVector, img_num_columns, img_num_rows, halfOfQubits);
    applyQFTVertically(xyVector, qftWorkSpace, img_num_columns, img_num_rows, halfOfQubits);
    fftshift2D(xyVector, img_num_rows, img_num_columns);
    printDeviceArray(xyVector, A_num_cols);

    CHECK_CUDA(cudaFree(xyVector))
    CHECK_CUDA(cudaFree(rThetaVector))
    CHECK_CUDA(cudaFree(qftWorkSpace))
    return EXIT_SUCCESS;
}

int runSys2()
{
    // Host problem definition
    int evenqubits = 12;
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
    // Vector
    cuDoubleComplex *rThetaVector; // theta slow, r fast
    cuDoubleComplex *xyVector = nullptr;
    CHECK_CUDA(cudaMallocManaged((void **)&rThetaVector, A_num_cols * sizeof(cuDoubleComplex)));

    cuDoubleComplex *qftWorkSpace;
    CHECK_CUDA(cudaMallocManaged((void **)&qftWorkSpace, img_num_rows * sizeof(cuDoubleComplex)));

    getData(rThetaVector, evenqubits, "../imageFile2.csv");

    // printDeviceArray(rThetaVector, A_num_cols);
    // fftshift2D(rThetaVector, img_num_rows, img_num_columns);
    // applyQFTVertically(rThetaVector,qftWorkSpace, img_num_columns, img_num_rows, halfOfQubits);
    // fftshift2D(rThetaVector, img_num_rows, img_num_columns);
    // printDeviceArray(rThetaVector, A_num_cols);

    printDeviceArray(rThetaVector, A_num_cols);
    Transpose(rThetaVector, img_num_rows, img_num_columns);
    CHECK_CUDA(static_cast<cudaError_t>(applyInterpolationMatrix(evenqubits, rThetaVector, xyVector)));
    printDeviceArray(xyVector, A_num_cols);

    // fftshift2D(xyVector, img_num_rows, img_num_columns);
    // applyQFTHorizontally(xyVector, img_num_columns, img_num_rows, halfOfQubits);
    // applyQFTVertically(xyVector, qftWorkSpace, img_num_columns, img_num_rows, halfOfQubits);
    // fftshift2D(xyVector, img_num_rows, img_num_columns);
    // printDeviceArray(xyVector, A_num_cols);

    CHECK_CUDA(cudaFree(xyVector))
    CHECK_CUDA(cudaFree(rThetaVector))
    CHECK_CUDA(cudaFree(qftWorkSpace))
    return EXIT_SUCCESS;
}


