#include <gtest/gtest.h>
#include "../CuQuantumControl/ApplyGates.hpp"
#include "../CudaControl/Helper.hpp"
#include <string>

class SetUpState : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialization logic
    }

    void TearDown() override
    {
        // Cleanup logic
    }
};

TEST(applyGates, X1_Base)
{
    const int nIndexBits = 1;
    const int nSvSize = (1 << nIndexBits);
    const int adjoint = false;

    std::array<cuDoubleComplex, nSvSize> input({{1, 0}, {0, 0}});
    std::array<cuDoubleComplex, nSvSize> output({{0, 0}, {1, 0}});
    int targets[] = {0};
    int control[] = {};

    cuDoubleComplex *d_sv;
    THROW_CUDA(cudaMallocManaged((void **)&d_sv, nSvSize * sizeof(cuDoubleComplex)));

    custatevecHandle_t handle = NULL;
    THROW_CUSTATEVECTOR(custatevecCreate(&handle));
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    mempcpy(d_sv, input.data(), nSvSize * sizeof(cuDoubleComplex));

    THROW_BROAD_ERROR(applyX(
        handle,
        nIndexBits,
        adjoint,
        targets,
        d_sv,
        extraWorkspace,
        extraWorkspaceSizeInBytes));

    if (extraWorkspace != nullptr)
        THROW_CUDA(cudaFree(extraWorkspace));
    THROW_CUDA(cudaFree(d_sv));
}