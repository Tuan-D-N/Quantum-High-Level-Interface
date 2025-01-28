#include <iostream>
#include <type_traits>
#include <cuComplex.h>
#include "CudaQControl/optimizingSystem.hpp"
#include "CuQuantumControl/ApplyGates.hpp"
#include "functionality_image/LoadImage.hpp"
#include "functionality_image/FilterAndLabel.hpp"

class circuitClass
{
private:
    cuComplex *dsv;
    int nSV;
    custatevecHandle_t handle;
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    const int adjoint = static_cast<int>(false);

    int m_blocks;
    int m_nQubits;

public:
    circuitClass(const int blocks, const int nQubits) : m_blocks(blocks), m_nQubits(nQubits)
    {
        nSV = 1 << nQubits;

        THROW_CUDA(cudaMallocManaged((void **)&dsv, nSV * sizeof(cuComplex)));
        THROW_CUSTATEVECTOR(custatevecCreate(&handle));
    }
    ~circuitClass()
    {
        ALERT_CUSTATEVECTOR(custatevecDestroy(handle));
        ALERT_CUDA(cudaFree(dsv));
    }
    void loadFloatIntoDsV(std::span<const float> inputState)
    {
        for (int i = 0; i < nSV; ++i)
        {
            dsv[i] = {inputState[i], 0};
        }
    }
    int getNumberOfParams() { return m_blocks * m_nQubits; }
    void RY(float RYParam, int target)
    {
        cuComplex mat[] = RYMatF(RYParam);
        THROW_BROAD_ERROR(applyGatesGeneral<precision::bit_32>(
            handle,
            m_nQubits,
            mat,
            adjoint,
            std::array<int, 1>{0},
            std::array<int, 0>{},
            dsv,
            extraWorkspace,
            extraWorkspaceSizeInBytes));
    }
    void CX(int target, int control)
    {
        THROW_BROAD_ERROR(applyX<precision::bit_32>(
            handle,
            m_nQubits,
            adjoint,
            std::array<int, 1>{target},
            std::array<int, 1>{control},
            dsv,
            extraWorkspace,
            extraWorkspaceSizeInBytes));
    }
    std::span<cuComplex> runCircuit(std::span<const float> inputState, const std::span<const double> weights)
    {
        loadFloatIntoDsV(inputState);

        for (int j = 0; j < m_blocks; ++j)
        {
            RY(weights[j * m_nQubits + 0], 0);
            for (int i = 0; i < m_nQubits; ++i)
            {
                RY(weights[j * m_nQubits + i], i);
                CX(i, i - 1);
            }
        }
        return std::span<cuComplex>(dsv, nSV);
    }
    std::function<std::span<cuComplex>(std::span<const float>, std::span<const double>)> getCircuitFunction()
    {
        return [this](std::span<const float> inputState, std::span<const double> weights) -> std::span<cuComplex>
        {
            return this -> runCircuit(inputState, weights);
        };
    }
};

int main()
{
    auto circuitOBJ = circuitClass(3, 10);

    const int INPUTWIDTH = 28, INPUTHEIGHT = 28;
    const int TARGETWIDTH = 32, TARGETHEIGHT = 32;

    std::vector<int> target_digits = {3, 6};

    auto [train_data, test_data] = load_mnist(
        "MNIST_Data/train-images.idx3-ubyte",
        "MNIST_Data/train-labels.idx1-ubyte",
        "MNIST_Data/t10k-images.idx3-ubyte",
        "MNIST_Data/t10k-labels.idx1-ubyte");

    auto &[x_train1, y_train1] = train_data;
    auto &[x_test1, y_test1] = test_data;

    auto filtered_train = filter_and_label(x_train1, y_train1, target_digits);
    auto filtered_test = filter_and_label(x_test1, y_test1, target_digits);

    auto &[x_train, y_train] = filtered_train;
    auto &[x_test, y_test] = filtered_test;

    auto optimizerOBJ = optimizingSystemBase(
        std::move(x_train),
        std::move(y_train),
        std::move(circuitOBJ.getCircuitFunction()));

    auto optimizer = cudaq::optimizers::adam();
    optimizer.step_size = 1;
    optimizer.max_eval = 100;
    optimizer.initial_parameters = std::vector<double>{-1.04636, -11.247, -28.1384, -25.2752, 9.91089, 43.9357, -15.5106, 5.88025, 10.6416, -9.45491, 24.5154, -33.4887, 6.31566, 18.5741, 9.52134, 8.17428, -3.59727, -36.4042, 0.688819, -23.659, -26.5258, 29.6948, -1.63063, 35.7586, 17.3797, 5.47379, -11.5945, 19.4702, 4.38039, 3.18327};


    auto result = optimizer.optimize(circuitOBJ.getNumberOfParams(), cudaq::optimizable_function(optimizerOBJ.getObjectiveFunction()));

    return 0;
}
