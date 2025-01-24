#include <gtest/gtest.h>
#include "../CuQuantumControl/Accessor.hpp"
#include "../CudaControl/Helper.hpp"
#include <string>

class AccessorGetTest : public ::testing::Test
{

protected:
    int m_nQubits;
    custatevecHandle_t handle = NULL;
    cuDoubleComplex *d_sv;
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;
    void SetUp() override
    {
        THROW_CUSTATEVECTOR(custatevecCreate(&handle));
    }

    void TearDown() override
    {
        if (extraWorkspace != nullptr)
            THROW_CUDA(cudaFree(extraWorkspace));
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
};

TEST_F(AccessorGetTest, oneMask1)
{
    const int nQubits = 3;
    setNIndex(nQubits);
    cuDoubleComplex input[] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}};
    cuDoubleComplex expectedOutput[] = {{2, 0}, {4, 0}, {6, 0}, {8, 0}};

    setState(input);
    int bitOrdering[] = {1, 2};
    int maskBitString[] = {1};
    int maskOrdering[] = {0};
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
        EXPECT_NEAR(d_sv[i].x, expectedOutput[i].x, 1e-9) << "Mismatch at index " << i << " in test Vector";
        EXPECT_NEAR(d_sv[i].y, expectedOutput[i].y, 1e-9) << "Mismatch at index " << i << " in test Vector";
    }
}