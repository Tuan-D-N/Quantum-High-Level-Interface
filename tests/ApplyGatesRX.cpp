#include <gtest/gtest.h>
#include "../CuQuantumControl/ApplyGates.hpp"
#include "../CudaControl/Helper.hpp"
#include <string>

class ApplyRXGates : public ::testing::Test
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

        THROW_BROAD_ERROR(applyRX(
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
            THROW_BROAD_ERROR(applyRX(
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
            EXPECT_NEAR(d_sv[i].x, expectedOutput[i].x, 1e-9) << "Mismatch at index " << i << " in test base";
            EXPECT_NEAR(d_sv[i].y, expectedOutput[i].y, 1e-9) << "Mismatch at index " << i << " in test base";
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

TEST_F(ApplyRXGates, RX_Base1)
{
    const int nQubits = 2;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{0.0707372, -1.99499}, {0.141474, -0.997495}, {0.212212, -3.98998}, {0.282949, -2.99248}};
    float p1 = 3;
    std::vector<int> targets = {0};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRXGates, RX_Base2)
{
    const int nQubits = 2;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{-0.989992, -0.28224}, {-1.97998, -0.14112}, {-2.96998, -0.56448}, {-3.95997, -0.42336}};
    float p1 = 3;
    std::vector<int> targets = {0, 0};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRXGates, RX_MultipleQubits)
{
    const int nQubits = 2;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{0.0707372, -2.99248}, {0.141474, -3.98998}, {0.212212, -0.997495}, {0.282949, -1.99499}};
    float p1 = 3;
    std::vector<int> targets = {1};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRXGates, RX_Controlled)
{
    const int nQubits = 2;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1., 0.}, {0.141474, -3.98998}, {3., 0.}, {0.282949, -1.99499}};
    float p1 = 3;
    std::vector<int> targets = {1};
    std::vector<int> controls = {0};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRXGates, RX_MultiControlled)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1., 0.}, {2., 0.}, {3., 0.}, {0.282949, -7.97996}, {5., 0.}, {6., 0.}, {7., 0.}, {0.565898, -3.98998}};
    float p1 = 3;
    std::vector<int> targets = {2};
    std::vector<int> controls = {0, 1};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRXGates, RX_MultipleTargets3)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{-1.19616, 7.89012}, {-1.12542, 6.89262}, {-1.05469, 5.89513}, {-0.98395, 4.89763}, {-0.913212, 3.90014}, {-0.842475, 2.90264}, {-0.771738, 1.90515}, {-0.701001, 0.907653}};
    float p1 = 3;
    std::vector<int> targets = {0, 1, 2};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRXGates, RX_MultipleTargets2)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{-6.95997, -0.56448}, {-7.94996, -0.7056}, {-4.95997, -0.56448}, {-5.94996, -0.7056}, {-2.95997, -0.56448}, {-3.94996, -0.7056}, {-0.95997, -0.56448}, {-1.94996, -0.7056}};
    float p1 = 3;
    std::vector<int> targets = {1, 2};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRXGates, RX_MultipleTargets1)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{-5.96497, -0.49392}, {-4.96497, -0.49392}, {-7.94496, -0.77616}, {-6.94496, -0.77616}, {-1.96497, -0.49392}, {-0.964974, -0.49392}, {-3.94496, -0.77616}, {-2.94496, -0.77616}};
    float p1 = 3;
    std::vector<int> targets = {0, 2};
    std::vector<int> controls = {};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRXGates, RX_MultipleTargets_SingleControls)
{
    const int nQubits = 3;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1., 0.}, {2., 0.}, {3., 0.}, {4., 0.}, {-7.93495, -0.91728}, {-6.93495, -0.91728}, {-5.93495, -0.91728}, {-4.93495, -0.91728}};
    float p1 = 3;
    std::vector<int> targets = {1, 0};
    std::vector<int> controls = {2};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}

TEST_F(ApplyRXGates, RX_MultipleTargets_MultipleControls)
{
    const int nQubits = 4;
    std::vector<cuDoubleComplex> input = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}, {13, 0}, {14, 0}, {15, 0}, {16, 0}};
    std::vector<cuDoubleComplex> expectedOutput = {{1., 0.}, {2., 0.}, {3., 0.}, {4., 0.}, {5., 0.}, {6., 0.}, {7., 0.}, {8., 0.}, {9., 0.}, {10., 0.}, {11., 0.}, {12., 0.}, {-15.8549, -2.04624}, {-14.8549, -2.04624}, {-13.8549, -2.04624}, {-12.8549, -2.04624}};
    float p1 = 3;
    std::vector<int> targets = {1, 0};
    std::vector<int> controls = {2, 3};

    runTest(input, expectedOutput, targets, controls, nQubits, p1);
}
