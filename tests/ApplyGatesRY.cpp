#include <gtest/gtest.h>
#include "../CuQuantumControl/ApplyGates.hpp"
#include "../CudaControl/Helper.hpp"
#include <string>

class ApplyRYGates : public ::testing::Test
{
private:
    void runTestVector(const std::vector<cuDoubleComplex> &inputState,
                       const std::vector<cuDoubleComplex> &expectedOutput,
                       const std::vector<int> &targets,
                       const std::vector<int> &controls,
                       int nQubits,
                       float p1,
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

        THROW_BROAD_ERROR(applyRY(
            handle,
            nQubits,
            adjoint,
            targets,
            controls,
            d_sv,
            extraWorkspace,
            extraWorkspaceSizeInBytes,
            p1));

        if (extraWorkspace != nullptr)
            THROW_CUDA(cudaFree(extraWorkspace));

        THROW_CUSTATEVECTOR(custatevecDestroy(handle));

        for (int i = 0; i < nSvSize; i++)
        {
            EXPECT_NEAR(d_sv[i].x, expectedOutput[i].x, 1e-4) << "Mismatch at index " << i << " in test Vector";
            EXPECT_NEAR(d_sv[i].y, expectedOutput[i].y, 1e-4) << "Mismatch at index " << i << " in test Vector";
        }
        THROW_CUDA(cudaFree(d_sv));
    }
    void runTestBase(const std::vector<cuDoubleComplex> &inputState,
                     const std::vector<cuDoubleComplex> &expectedOutput,
                     const std::vector<int> &targets,
                     const std::vector<int> &controls,
                     int nQubits,
                     float p1,
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
            THROW_BROAD_ERROR(applyRY(
                handle,
                nQubits,
                adjoint,
                target,
                controls.data(),
                controls.size(),
                d_sv,
                extraWorkspace,
                extraWorkspaceSizeInBytes,
                p1));
        }

        if (extraWorkspace != nullptr)
            THROW_CUDA(cudaFree(extraWorkspace));

        THROW_CUSTATEVECTOR(custatevecDestroy(handle));

        for (int i = 0; i < nSvSize; i++)
        {
            EXPECT_NEAR(d_sv[i].x, expectedOutput[i].x, 1e-4) << "Mismatch at index " << i << " in test base";
            EXPECT_NEAR(d_sv[i].y, expectedOutput[i].y, 1e-4) << "Mismatch at index " << i << " in test base";
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
        float p1,
        int adjoint = false)
    {
        runTestVector(inputState, expectedOutput, targets, controls, nQubits, p1, adjoint);
        runTestBase(inputState, expectedOutput, targets, controls, nQubits, p1, adjoint);
    }
};

TEST_F(ApplyRYGates, RY_Base1)
{
    const int nQubits = 2;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{-1.92425, 0.}, {1.13897, 0.}, {-3.77777, 0.}, {3.27543, 0.}};
    float p1 = 3;
    std::vector<int> targets = {0};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRYGates, RY_Base2)
{
    const int nQubits = 2;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{-1.27223, 0.}, {-1.83886, 0.}, {-3.53446, 0.}, {-3.53661, 0.}};
    float p1 = 3;
    std::vector<int> targets = {0, 0};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRYGates, RY_MultipleQubits)
{
    const int nQubits = 2;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{-2.92175, 0.}, {-3.84851, 0.}, {1.20971, 0.}, {2.27794, 0.}};
    float p1 = 3;
    std::vector<int> targets = {1};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRYGates, RY_Controlled)
{
    const int nQubits = 2;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1., 0.}, {-3.84851, 0.}, {3., 0.}, {2.27794, 0.}};
    float p1 = 3;
    std::vector<int> targets = {1};
    std::vector<int> controls = {0};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRYGates, RY_MultiControlled)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1., 0.}, {2., 0.}, {3., 0.}, {-7.69701, 0.}, {5., 0.}, {6., 0.}, {7., 0.}, {4.55588, 0.}};
    float p1 = 3;
    std::vector<int> targets = {2};
    std::vector<int> controls = {0, 1};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRYGates, RY_MultipleTargets3)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{-6.79307, 0.}, {6.90331, 0.}, {5.97656, 0.}, {-5.82068, 0.}, {4.12304, 0.}, {-3.68421, 0.}, {-2.61598, 0.}, {1.78402, 0.}};
    float p1 = 3;
    std::vector<int> targets = {0, 1, 2};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRYGates, RY_MultipleTargets2)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{3.63219, 0.}, {-3.18666, 0.}, {-2.18666, 0.}, {1.36781, 0.}, {7.06771, 0.}, {-7.14663, 0.}, {-6.14663, 0.}, {5.93229, 0.}};
    float p1 = 3;
    std::vector<int> targets = {1, 2};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRYGates, RY_MultipleTargets1)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{5.48106, 0.}, {-5.31777, 0.}, {7.19882, 0.}, {-7.29776, 0.}, {-2.31777, 0.}, {1.51894, 0.}, {-4.29776, 0.}, {3.80118, 0.}};
    float p1 = 3;
    std::vector<int> targets = {0, 2};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRYGates, RY_MultipleTargets_SingleControls)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{3.63219, 0.}, {-3.18666, 0.}, {-2.18666, 0.}, {1.36781, 0.}, {7.06771, 0.}, {-7.14663, 0.}, {-6.14663, 0.}, {5.93229, 0.}};
    float p1 = 3;
    std::vector<int> targets = {1, 0};
    std::vector<int> controls = {2};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRYGates, RY_MultipleTargets_MultipleControls)
{
    const int nQubits = 4;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}, {13, 0}, {14, 0}, {15, 0}, {16, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1., 0.}, {2., 0.}, {3., 0.}, {4., 0.}, {5., 0.}, {6., 0.}, {7., 0.}, {8., 0.}, {9., 0.}, {10., 0.}, {11., 0.}, {12., 0.}, {13.9387, 0.}, {-15.0666, 0.}, {-14.0666, 0.}, {15.0613, 0.}};
    float p1 = 3;
    std::vector<int> targets = {1, 0};
    std::vector<int> controls = {2, 3};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}
