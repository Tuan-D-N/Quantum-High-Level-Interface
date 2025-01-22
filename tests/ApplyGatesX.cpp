#include <gtest/gtest.h>
#include "../CuQuantumControl/ApplyGates.hpp"
#include "../CudaControl/Helper.hpp"
#include <string>

class ApplyXTest : public ::testing::Test
{
private:
    void runTestVector(const std::vector<cuDoubleComplex> &inputState,
                       const std::vector<cuDoubleComplex> &expectedOutput,
                       const std::vector<int> &targets,
                       const std::vector<int> &controls,
                       int nQubits,
                       int adjoint = false)
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

        THROW_BROAD_ERROR(applyX(
            handle,
            nQubits,
            adjoint,
            targets,
            controls,
            d_sv,
            extraWorkspace,
            extraWorkspaceSizeInBytes));

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
                     const std::vector<int> &targets,
                     const std::vector<int> &controls,
                     int nQubits,
                     int adjoint = false)
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

        for (int target : targets)
        {
            THROW_BROAD_ERROR(applyX(
                handle,
                nQubits,
                adjoint,
                target,
                controls.data(),
                controls.size(),
                d_sv,
                extraWorkspace,
                extraWorkspaceSizeInBytes));
        }

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

    void runTest(
        const std::vector<cuDoubleComplex> &inputState,
        const std::vector<cuDoubleComplex> &expectedOutput,
        const std::vector<int> &targets,
        const std::vector<int> &controls,
        int nQubits,
        int adjoint = false)
    {
        runTestVector(inputState, expectedOutput, targets, controls, nQubits, adjoint);
        runTestBase(inputState, expectedOutput, targets, controls, nQubits, adjoint);
    }
};

TEST_F(ApplyXTest, X_Base1)
{
    const int nQubits = 2;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{2, 0}, {1, 0}, {4, 0}, {3, 0}};
    std::vector<int> targets = {0};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits);
}

TEST_F(ApplyXTest, X_Base2)
{
    const int nQubits = 2;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    std::vector<int> targets = {0, 0};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits);
}

TEST_F(ApplyXTest, X_MultipleQubits)
{
    const int nQubits = 2;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{3, 0}, {4, 0}, {1, 0}, {2, 0}};
    std::vector<int> targets = {1};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits);
}

TEST_F(ApplyXTest, X_Controlled)
{
    const int nQubits = 2;
    std::vector<cuDoubleComplex> input = {{1, 0}, {0, 0}, {0, 0}, {0, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1, 0}, {0, 0}, {0, 0}, {0, 0}};
    std::vector<int> targets = {1};
    std::vector<int> controls = {0};

    runTest(input, expectedOutput, targets, controls, nQubits);
}

TEST_F(ApplyXTest, X_MultiControlled)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1, 0}, {2, 0}, {3, 0}, {8, 0}, {5, 0}, {6, 0}, {7, 0}, {4, 0}};
    std::vector<int> targets = {2};
    std::vector<int> controls = {0, 1};

    runTest(input, expectedOutput, targets, controls, nQubits);
}

TEST_F(ApplyXTest, X_MultipleTargets3)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{8, 0}, {7, 0}, {6, 0}, {5, 0}, {4, 0}, {3, 0}, {2, 0}, {1, 0}};
    std::vector<int> targets = {0, 1, 2};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits);
}

TEST_F(ApplyXTest, X_MultipleTargets2)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{7, 0}, {8, 0}, {5, 0}, {6, 0}, {3, 0}, {4, 0}, {1, 0}, {2, 0}};
    std::vector<int> targets = {1, 2};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits);
}

TEST_F(ApplyXTest, X_MultipleTargets1)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{6, 0}, {5, 0}, {8, 0}, {7, 0}, {2, 0}, {1, 0}, {4, 0}, {3, 0}};
    std::vector<int> targets = {0, 2};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits);
}

TEST_F(ApplyXTest, X_MultipleTargets_SingleControls)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {8, 0}, {7, 0}, {6, 0}, {5, 0}};
    std::vector<int> targets = {1, 0};
    std::vector<int> controls = {2};

    runTest(input, expectedOutput, targets, controls, nQubits);
}

TEST_F(ApplyXTest, X_MultipleTargets_MultipleControls)
{
    const int nQubits = 4;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}, {13, 0}, {14, 0}, {15, 0}, {16, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}, {16, 0}, {15, 0}, {14, 0}, {13, 0}};
    std::vector<int> targets = {1, 0};
    std::vector<int> controls = {2, 3};

    runTest(input, expectedOutput, targets, controls, nQubits);
}
