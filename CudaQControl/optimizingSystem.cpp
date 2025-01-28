#include "optimizingSystem.hpp"


optimizingSystemBase::optimizingSystemBase(
    const std::vector<std::vector<float>> &x_data,
    const std::vector<int> &y_labels,
    circuitFunctionParams &circuit)
    : m_x_data(x_data), m_y_labels(y_labels), m_circuit(circuit)
{
}

optimizingSystemBase::optimizingSystemBase(
    std::vector<std::vector<float>> &&x_data,
    std::vector<int> &&y_labels,
    circuitFunctionParams &circuit)
    : m_x_data(std::move(x_data)), m_y_labels(std::move(y_labels)), m_circuit(circuit)
{
}

double optimizingSystemBase::lossFunction(const std::vector<double> &paramsVector)
{
    double totalLoss = 0;

    for (int i = 0; i < m_x_data.size(); ++i)
    {
        const auto label = m_y_labels[i];

        const std::vector<float> in_SV = m_x_data[i]; // Make a copy

        auto result = m_circuit(in_SV, paramsVector);

        auto binaryResult = measure1QubitUnified<precision::bit_32>(result);

        if (label == 0)
        {
            totalLoss -= std::log(std::abs(binaryResult.first + 0.01));
        }
        else if (label == 1)
        {
            totalLoss -= std::log(std::abs(binaryResult.second + 0.01));
        }
        else
        {
            std::cerr << "label of number: " << label << " was shown.";
            throw std::logic_error("Expected only 2 label types");
        }
    }

    return totalLoss;
}

double optimizingSystemBase::objectiveFunction(const std::vector<double> &inputVector, std::vector<double> &gradient)
{
    double resultAtX = lossFunction(inputVector);
    auto gradientFunction = cudaq::gradients::central_difference();
    auto lossFunctionCallable = [this](const std::vector<double> &inputVector)
    {
        return this->lossFunction(inputVector);
    };
    gradient = gradientFunction.compute(inputVector, lossFunctionCallable, resultAtX);
    return resultAtX;
}
