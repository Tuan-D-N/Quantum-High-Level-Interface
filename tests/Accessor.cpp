#include <gtest/gtest.h>
#include "../CuQuantumControl/Accessor.hpp"
#include "../CudaControl/Helper.hpp"
#include <string>

class AccessorGetTest : public ::testing::Test
{

protected:
    int m_nQubits;
    custatevecHandle_t handle = NULL;
    cuDoubleComplex *d_sv = nullptr;
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;
    void SetUp() override
    {
        THROW_CUSTATEVECTOR(custatevecCreate(&handle));
    }

    void TearDown() override
    {
        if (extraWorkspace != nullptr)
        {
            std::cout << "Wordspace was freed\n";
            THROW_CUDA(cudaFree(extraWorkspace));
        }
        THROW_CUSTATEVECTOR(custatevecDestroy(handle));
        THROW_CUDA(cudaFree(d_sv));
    }
    void setNIndex(const int nQubits)
    {
        m_nQubits = nQubits;
        THROW_CUDA(cudaMallocManaged((void **)&d_sv, nSvSize() * sizeof(cuDoubleComplex)));
    }
    void setState(std::span<cuDoubleComplex> inputState)
    {
        memcpy(d_sv, inputState.data(), nSvSize() * sizeof(cuDoubleComplex));
    }
    int nSvSize()
    {
        return 1 << m_nQubits;
    }
    inline void runCheck(
        const int nQubits,
        std::span<cuDoubleComplex> input,
        std::span<cuDoubleComplex> expectedOutput,
        std::span<int> bitOrdering,
        std::span<int> maskBitString,
        std::span<int> maskOrdering)
    {
        setNIndex(nQubits);
        setState(input);
        std::vector<cuDoubleComplex> outBuffer(1 << (nQubits - std::span(maskOrdering).size()));
        const int out_begin = 0;
        const int out_end = outBuffer.size();
        THROW_BROAD_ERROR(
            applyAccessorGet(
                handle,
                m_nQubits,
                bitOrdering,
                maskBitString,
                maskOrdering,
                d_sv,
                outBuffer,
                extraWorkspace,
                extraWorkspaceSizeInBytes));

        THROW_CUDA(cudaDeviceSynchronize());

        EXPECT_EQ(outBuffer.size(), std::span(expectedOutput).size());

        for (int i = 0; i < outBuffer.size(); ++i)
        {
            EXPECT_NEAR(outBuffer[i].x, expectedOutput[i].x, 1e-9) << "Mismatch at index " << i << " in test Vector";
            EXPECT_NEAR(outBuffer[i].y, expectedOutput[i].y, 1e-9) << "Mismatch at index " << i << " in test Vector";
        }
    }
};

TEST_F(AccessorGetTest, oneMask1)
{
    const int nQubits = 3;
    cuDoubleComplex input[] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    cuDoubleComplex expectedOutput[] = {{2, 0}, {4, 0}, {6, 0}, {8, 0}};
    int bitOrdering[] = {1, 2};
    int maskBitString[] = {1};
    int maskOrdering[] = {0};
    runCheck(nQubits, input, expectedOutput, bitOrdering, maskBitString, maskOrdering);
}

TEST_F(AccessorGetTest, oneMask2)
{
    const int nQubits = 3;
    cuDoubleComplex input[] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    cuDoubleComplex expectedOutput[] = {{5, 0}, {6, 0}, {7, 0}, {8, 0}};
    int bitOrdering[] = {0, 1};
    int maskBitString[] = {1};
    int maskOrdering[] = {2};
    runCheck(nQubits, input, expectedOutput, bitOrdering, maskBitString, maskOrdering);
}

TEST_F(AccessorGetTest, oneMask3)
{
    const int nQubits = 3;
    cuDoubleComplex input[] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    cuDoubleComplex expectedOutput[] = {{1, 0}, {2, 0}, {5, 0}, {6, 0}};
    int bitOrdering[] = {0, 2};
    int maskBitString[] = {0};
    int maskOrdering[] = {1};
    runCheck(nQubits, input, expectedOutput, bitOrdering, maskBitString, maskOrdering);
}

TEST_F(AccessorGetTest, oneMask4)
{
    const int nQubits = 3;
    cuDoubleComplex input[] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    cuDoubleComplex expectedOutput[] = {{2, 0}, {6, 0}, {4, 0}, {8, 0}};
    int bitOrdering[] = {2, 1};
    int maskBitString[] = {1};
    int maskOrdering[] = {0};
    runCheck(nQubits, input, expectedOutput, bitOrdering, maskBitString, maskOrdering);
}