#include <gtest/gtest.h>
#include "../CuQuantumControl/SwapGates.hpp"
#include "../CudaControl/Helper.hpp"
#include <string>

class ApplySwapGates : public ::testing::Test
{
private:
    void runTestVector(const std::vector<cuDoubleComplex> &inputState,
                       const std::vector<cuDoubleComplex> &expectedOutput,
                       const std::span<const int2> &bitSwaps,
                       int nQubits)
    {
        const int nSvSize = (1 << nQubits);

        ASSERT_EQ(inputState.size(), nSvSize);
        ASSERT_EQ(expectedOutput.size(), nSvSize);

        cuDoubleComplex *d_sv;
        THROW_CUDA(cudaMallocManaged((void **)&d_sv, nSvSize * sizeof(cuDoubleComplex)));

        custatevecHandle_t handle = NULL;
        THROW_CUSTATEVECTOR(custatevecCreate(&handle));

        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        memcpy(d_sv, inputState.data(), nSvSize * sizeof(cuDoubleComplex));

        applySwap<precision::bit_64>(handle, nQubits, bitSwaps, d_sv);

        if (extraWorkspace != nullptr)
            THROW_CUDA(cudaFree(extraWorkspace));

        THROW_CUSTATEVECTOR(custatevecDestroy(handle));

        for (int i = 0; i < nSvSize; i++)
        {
            EXPECT_NEAR(d_sv[i].x, expectedOutput[i].x, 1e-9) << "Mismatch at index " << i << " in test Vector";
            EXPECT_NEAR(d_sv[i].y, expectedOutput[i].y, 1e-9) << "Mismatch at index " << i << " in test Vector";
        }
        THROW_CUDA(cudaFree(d_sv));
    }
    void runTestBase(const std::vector<cuDoubleComplex> &inputState,
                     const std::vector<cuDoubleComplex> &expectedOutput,
                     const std::span<const int2> &bitSwaps,
                     int nQubits)
    {
        const int nSvSize = (1 << nQubits);

        ASSERT_EQ(inputState.size(), nSvSize);
        ASSERT_EQ(expectedOutput.size(), nSvSize);

        cuDoubleComplex *d_sv;
        THROW_CUDA(cudaMallocManaged((void **)&d_sv, nSvSize * sizeof(cuDoubleComplex)));

        custatevecHandle_t handle = NULL;
        THROW_CUSTATEVECTOR(custatevecCreate(&handle));

        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        memcpy(d_sv, inputState.data(), nSvSize * sizeof(cuDoubleComplex));

        applySwap<precision::bit_64>(handle, nQubits, bitSwaps.data(), bitSwaps.size(), d_sv);

        if (extraWorkspace != nullptr)
            THROW_CUDA(cudaFree(extraWorkspace));

        THROW_CUSTATEVECTOR(custatevecDestroy(handle));

        for (int i = 0; i < nSvSize; i++)
        {
            EXPECT_NEAR(d_sv[i].x, expectedOutput[i].x, 1e-9) << "Mismatch at index " << i << " in test Vector";
            EXPECT_NEAR(d_sv[i].y, expectedOutput[i].y, 1e-9) << "Mismatch at index " << i << " in test Vector";
        }
        THROW_CUDA(cudaFree(d_sv));
    }

protected:
    void SetUp() override {}

    void TearDown() override {}

    void runTest(const std::vector<cuDoubleComplex> &inputState,
                 const std::vector<cuDoubleComplex> &expectedOutput,
                 const std::span<const int2> &bitSwaps,
                 int nQubits)
    {
        runTestVector(inputState, expectedOutput, bitSwaps, nQubits);
        runTestBase(inputState, expectedOutput, bitSwaps, nQubits);
    }
};

TEST_F(ApplySwapGates, SwapOnce2Qubits)
{
    const int nQubits = 2;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1, 0}, {3, 0}, {2, 0}, {4, 0}};
    int2 bitSwaps[] = {{0, 1}};

    runTest(input, expectedOutput, bitSwaps, nQubits);
}

TEST_F(ApplySwapGates, SwapOnce4Qubits)
{
    const int nQubits = 4;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}, {13, 0}, {14, 0}, {15, 0}, {16, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {13, 0}, {14, 0}, {15, 0}, {16, 0}};
    int2 bitSwaps[] = {{3, 2}};

    runTest(input, expectedOutput, bitSwaps, nQubits);
}
TEST_F(ApplySwapGates, SwapTwice4Qubits)
{
    const int nQubits = 4;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}, {13, 0}, {14, 0}, {15, 0}, {16, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1, 0}, {5, 0}, {9, 0}, {13, 0}, {2, 0}, {6, 0}, {10, 0}, {14, 0}, {3, 0}, {7, 0}, {11, 0}, {15, 0}, {4, 0}, {8, 0}, {12, 0}, {16, 0}};
    int2 bitSwaps[] = {{3, 1}, {2, 0}};

    runTest(input, expectedOutput, bitSwaps, nQubits);
}
TEST_F(ApplySwapGates, SwapTwice4Qubits4)
{
    const int nQubits = 4;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}, {13, 0}, {14, 0}, {15, 0}, {16, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1, 0}, {9, 0}, {5, 0}, {13, 0}, {3, 0}, {11, 0}, {7, 0}, {15, 0}, {2, 0}, {10, 0}, {6, 0}, {14, 0}, {4, 0}, {12, 0}, {8, 0}, {16, 0}};
    int2 bitSwaps[] = {{3, 0}, {2, 1}};

    runTest(input, expectedOutput, bitSwaps, nQubits);
}
