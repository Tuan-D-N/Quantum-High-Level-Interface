#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include "custatevec.h" // Include cuStateVec headers
#include "../CuQuantumControl/QftStateVec.hpp"
#include "../CuQuantumControl/helper.hpp"
#include <cstring>

// Test 1: Verify success of ApplyQFTOnStateVector
TEST(QFTTests, ApplyQFTSuccess)
{
    const int nQubits = 3;
    const int nSvSize = 1 << nQubits;

    cuDoubleComplex *d_stateVector;

    const auto err = cudaMallocManaged((void **)&d_stateVector, nSvSize * sizeof(cuDoubleComplex));
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    int result = ApplyQFTOnStateVector(d_stateVector, nQubits);

    cudaFree(d_stateVector);
    EXPECT_EQ(result, cudaSuccess);
}

// Test 2: Verify state vector modification
TEST(QFTTests, VerifyStateVector)
{
    const int nQubits = 3;
    const int nSvSize = 1 << nQubits;

    cuDoubleComplex h_stateVector[] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    cuDoubleComplex h_stateVectorResult[] = {{12.7279, 0.0}, {-1.41421, -3.41421}, {-1.41421, -1.41421}, {-1.41421, -0.58579}, {-1.41421, 0.0}, {-1.41421, 0.58579}, {-1.41421, 1.41421}, {-1.41421, 3.41421}};
    cuDoubleComplex *d_stateVector;

    const auto err = cudaMallocManaged((void **)&d_stateVector, nSvSize * sizeof(cuDoubleComplex));
    std::memcpy(d_stateVector, h_stateVector, nQubits);

    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    ApplyQFTOnStateVector(d_stateVector, nQubits);

    for (int i = 0; i < nSvSize; ++i)
    {
        EXPECT_NEAR(d_stateVector[i].x, h_stateVectorResult[i].x, 0.001);
        EXPECT_NEAR(d_stateVector[i].y, h_stateVectorResult[i].y, 0.001);
    }

    cudaFree(d_stateVector);
}

// Test 3: Edge case - Single qubit
TEST(QFTTests, SingleQubitQFT)
{
    const int nQubits = 1;
    const int nSvSize = 1 << nQubits;

    cuDoubleComplex h_stateVector[] = {{1, 1}, {8, 1}};
    cuDoubleComplex h_stateVectorResult[] = {{6.36396, 1.41421}, {-4.94975, 0.0}};
    cuDoubleComplex *d_stateVector;

    const auto err = cudaMallocManaged((void **)&d_stateVector, nSvSize * sizeof(cuDoubleComplex));
    std::memcpy(d_stateVector, h_stateVector, nQubits);

    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    ApplyQFTOnStateVector(d_stateVector, nQubits);

    for (int i = 0; i < nSvSize; ++i)
    {
        EXPECT_NEAR(d_stateVector[i].x, h_stateVectorResult[i].x, 0.001);
        EXPECT_NEAR(d_stateVector[i].y, h_stateVectorResult[i].y, 0.001);
    }

    cudaFree(d_stateVector);
}

// Test 4: Edge case - Large qubit count
TEST(QFTTests, LargeQubitCount10)
{
    int largeQubitCount = 10;
    int largeStateVectorSize = 1 << largeQubitCount;

    std::vector<cuDoubleComplex> h_stateVector(largeQubitCount, make_cuDoubleComplex(0, 0));
    h_stateVector[0] = make_cuDoubleComplex(1, 0);

    cuDoubleComplex *d_largeStateVector;
    const auto err = cudaMallocManaged((void **)&d_largeStateVector, largeStateVectorSize * sizeof(cuDoubleComplex));
    std::memcpy(d_largeStateVector, h_stateVector.data(), largeStateVectorSize);
    int result = ApplyQFTOnStateVector(d_largeStateVector, largeQubitCount);

    EXPECT_EQ(result, cudaSuccess);

    for (int i = 0; i < largeStateVectorSize; ++i)
    {
        EXPECT_NEAR(d_largeStateVector[i].x, d_largeStateVector[0].x, 0.001);
        EXPECT_NEAR(d_largeStateVector[i].y, d_largeStateVector[0].y, 0.001);
    }

    cudaFree(d_largeStateVector);
}
