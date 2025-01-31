#pragma once
#include <iostream>
#include <functional>
#include <vector>
#include "../CuQuantumControl/stateContracter.hpp"

class optimizingSystem
{
private:
public:
    optimizingSystem() = default;
    ~optimizingSystem() = default;

    /// @brief This is the function that determines the terrain. This function need to be
    /// implemented with the training data, labels etc.
    /// @param paramsVector This is the input parameters
    /// @return This is the loss value, lower is better.
    virtual double lossFunction(const std::vector<double> &paramsVector) = 0;

    virtual double objectiveFunction(const std::vector<double> &inputVector, std::vector<double> &gradient) = 0;

    virtual double accuracyTrain(const std::vector<double> &paramsVector) = 0;

    virtual double accuracyTest(const std::vector<double> &paramsVector) = 0;

    /// @brief Returns a pointer to the objective function
    /// @return Pointer to the objective function
    std::function<double(const std::vector<double> &, std::vector<double> &)> getObjectiveFunction();
};

class optimizingSystemBase : public optimizingSystem
{
protected:
    using circuitFunctionParams = std::function<std::span<cuComplex>(std::span<const float>, const std::span<const double>)>; // PARAMs: input Vector, input vector
    std::vector<std::vector<float>> m_x_data_train;
    std::vector<std::vector<float>> m_y_label_train;
    std::vector<std::vector<float>> m_x_data_test;
    std::vector<std::vector<float>> m_y_label_test;
    circuitFunctionParams m_circuit;

public:
    optimizingSystemBase(const std::vector<std::vector<float>> &x_data_train,
                         const std::vector<std::vector<float>> &y_labels_train,
                         const std::vector<std::vector<float>> &x_data_test,
                         const std::vector<std::vector<float>> &y_labels_test,
                         circuitFunctionParams &circuit);

    optimizingSystemBase(std::vector<std::vector<float>> &&x_data_train,
                         std::vector<std::vector<float>> &&y_labels_train,
                         std::vector<std::vector<float>> &&x_data_test,
                         std::vector<std::vector<float>> &&y_labels_test,
                         circuitFunctionParams &&circuit);

    ~optimizingSystemBase() = default;

    virtual double lossFunction(const std::vector<double> &paramsVector) override;

    virtual double objectiveFunction(const std::vector<double> &inputVector, std::vector<double> &gradient) override;

    virtual double accuracyTrain(const std::vector<double> &paramsVector) override;

    virtual double accuracyTest(const std::vector<double> &paramsVector) override;
};
