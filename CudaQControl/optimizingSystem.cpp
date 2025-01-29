#include <algorithm>
#include "optimizingSystem.hpp"
#include <functional>
#include <iostream>
#include <vector>
#include <cudaq/algorithm.h>
#include <cudaq/optimizers.h>
#include "stateContracter.hpp"

std::function<double(const std::vector<double> &, std::vector<double> &)> optimizingSystem::getObjectiveFunction()
{
    return [this](const std::vector<double> &inputVector, std::vector<double> &gradient)
    {
        return this->objectiveFunction(inputVector, gradient);
    };
}

optimizingSystemBase::optimizingSystemBase(
    const std::vector<std::vector<float>> &x_data_train,
    const std::vector<std::vector<float>> &y_labels_train,
    const std::vector<std::vector<float>> &x_data_test,
    const std::vector<std::vector<float>> &y_labels_test,
    circuitFunctionParams &circuit)
    : m_x_data_train(x_data_train),
      m_y_label_train(y_labels_train),
      m_circuit(circuit),
      m_x_data_test(x_data_test),
      m_y_label_test(y_labels_test)
{
    assert(m_y_label_test.size() > 0);
    assert(m_y_label_train.size() > 0);
    assert(m_y_label_test[0].size() == 2);
    assert(m_y_label_train[0].size() == 2);
}

optimizingSystemBase::optimizingSystemBase(
    std::vector<std::vector<float>> &&x_data_train,
    std::vector<std::vector<float>> &&y_labels_train,
    std::vector<std::vector<float>> &&x_data_test,
    std::vector<std::vector<float>> &&y_labels_test,
    circuitFunctionParams &&circuit)
    : m_x_data_train(std::move(x_data_train)),
      m_y_label_train(std::move(y_labels_train)),
      m_circuit(std::move(circuit)),
      m_x_data_test(std::move(x_data_test)),
      m_y_label_test(std::move(y_labels_test))
{
    assert(m_y_label_test.size() > 0);
    assert(m_y_label_train.size() > 0);
    assert(m_y_label_test[0].size() == 2);
    assert(m_y_label_train[0].size() == 2);
}

double optimizingSystemBase::lossFunction(const std::vector<double> &paramsVector)
{
    double totalLoss = 0;

    for (int train_index = 0; train_index < m_x_data_train.size(); ++train_index)
    {
        const auto label = m_y_label_train[train_index];

        const std::vector<float> &in_SV = m_x_data_train[train_index];

        auto result = m_circuit(in_SV, paramsVector);

        auto binaryResult = measure1QubitUnified<precision::bit_32>(result); // measures qubits

        for (int label_index = 0; label_index < binaryResult.size(); ++label_index)
        {
            double diff = binaryResult[label_index] * m_x_data_test[train_index][label_index];
            totalLoss -= std::log(std::abs(diff + 0.01));
        }
    }

    return totalLoss;
}

double optimizingSystemBase::objectiveFunction(const std::vector<double> &inputVector, std::vector<double> &gradient)
{
    static int iter = 0;
    ++iter;
    double resultAtX = lossFunction(inputVector);
    auto gradientFunction = cudaq::gradients::central_difference();
    gradientFunction.step = 0.01;
    auto lossFunctionCallable = [this](const std::vector<double> &inputVector)
    {
        return this->lossFunction(inputVector);
    };
    gradient = gradientFunction.compute(inputVector, lossFunctionCallable, resultAtX);

    std::cout << "params: ";
    for (auto param : inputVector)
    {
        std::cout << param << " , ";
    }
    std::cout << "\n";

    std::cout << "gradient: ";
    for (auto param : gradient)
    {
        std::cout << param << " , ";
    }
    std::cout << "\n";
    std::cout << "Looped " << iter << ": " << resultAtX << "\n ";
    std::cout << "Accuracy train" << ": " << accuracyTrain(inputVector) << "\n ";
    std::cout << "Accuracy test" << ": " << accuracyTest(inputVector) << "\n ";
    std::cout << "\n\n\n\n";

    return resultAtX;
}

double optimizingSystemBase::accuracyTrain(const std::vector<double> &paramsVector)
{
    double totalCorrect = 0;
    double totalResult = 0;

    for (int i = 0; i < m_x_data_train.size(); ++i)
    {
        const auto label_one_hot_shot = m_y_label_train[i];
        const std::vector<float>& in_SV = m_x_data_train[i]; 
        auto result = m_circuit(in_SV, paramsVector);
        auto binaryResult = measure1QubitUnified<precision::bit_32>(result);

        ++totalResult;

        //Gets the index of the highest element in the array
        auto &arr = binaryResult;
        auto indexOfMaxElement = std::distance(arr.begin(), std::max_element(arr.begin(), arr.end()));


        totalCorrect += label_one_hot_shot[indexOfMaxElement];
    }

    return totalCorrect / totalResult;
}

double optimizingSystemBase::accuracyTest(const std::vector<double> &paramsVector)
{
    double totalCorrect = 0;
    double totalResult = 0;

    for (int i = 0; i < m_x_data_test.size(); ++i)
    {
        const auto label_one_hot_shot = m_y_label_test[i];
        const std::vector<float>& in_SV = m_x_data_test[i];
        auto result = m_circuit(in_SV, paramsVector);
        auto binaryResult = measure1QubitUnified<precision::bit_32>(result);

        ++totalResult;

        //Gets the index of the highest element in the array
        auto &arr = binaryResult;
        auto indexOfMaxElement = std::distance(arr.begin(), std::max_element(arr.begin(), arr.end()));


        totalCorrect += label_one_hot_shot[indexOfMaxElement];
    }

    return totalCorrect / totalResult;
}
