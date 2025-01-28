#pragma once
#include <functional>
#include <vector>
#include <cudaq/algorithm.h>
#include <cudaq/optimizers.h>
#include "stateContracter.hpp"

class optimizingSystem
{
private:
public:
    optimizingSystem() = default;
    ~optimizingSystem() = default;

    /// @brief This is the function that determines the terrain. This function need to be
    /// implemented with the training data, labels etc.
    /// @param inputVector This is the input parameters
    /// @return This is the loss value, lower is better.
    virtual double lossFunction(const std::vector<double> &inputVector) = 0;

    virtual double objectiveFunction(const std::vector<double> &inputVector, std::vector<double> &gradient) = 0;
};

class optimizingSystemBase : optimizingSystem
{
protected:
    using circuitFunctionParams = std::function<std::span<cuComplex>(std::span<const float>, const std::span<const double>)>; // PARAMs: input Vector, input vector
    std::vector<std::vector<float>> m_x_data;
    std::vector<int> m_y_labels;
    circuitFunctionParams &m_circuit;

public:
    optimizingSystemBase(const std::vector<std::vector<float>> &x_data,
                         const std::vector<int> &y_labels,
                         circuitFunctionParams &circuit);

    optimizingSystemBase(std::vector<std::vector<float>> &&x_data,
                         std::vector<int> &&y_labels,
                         circuitFunctionParams &circuit);

    ~optimizingSystemBase() = default;

    virtual double lossFunction(const std::vector<double> &inputVector) override;

    virtual double objectiveFunction(const std::vector<double> &inputVector, std::vector<double> &gradient) override;
};
