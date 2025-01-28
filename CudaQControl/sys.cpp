#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/gradients.h>
#include <cudaq/optimizers.h>
#include <iostream>

// double terrain(const std::vector<double> &inputVector)
// {
//     // std::cout << "params: ";
//     // for (auto param : inputVector)
//     // {
//     //     std::cout << param << " , ";
//     // }
//     // std::cout << "\n";

//     double total = 0;
//     for (int i = 0; i < x_train.size(); ++i)
//     {
//         auto &input_picture = x_train[i];
//         auto result = cudaq::get_state(circuit{}, input_picture, inputVector);

//         for (int j = 0; j < targetStates.size(); ++j)
//         {
//             total -= y_train_one_hot[i][j] * std::log(std::abs(result.amplitude(targetStates[j])) + 0.01);
//         }

//         total -= y_train_one_hot[i][0] * std::log(std::abs(result.amplitude(zeroState)) + 0.01);
//         total -= y_train_one_hot[i][1] * std::log(std::abs(result.amplitude(oneState)) + 0.01);
//         // std::cout << "Inside loop " << i << ": " << total << " , " << y_train_one_hot[i][0] << " , " << y_train_one_hot[i][1] << " , " << result.amplitude(zeroState) << " , " << result.amplitude(oneState) << "\n";
//     }
//     // std::cout << "Looped once: " << total << "\n";
//     return total;
// }

// double objectiveFunction(const std::vector<double> &inputVector, std::vector<double> &gradient)
// {

//     double resultAtX = terrain(inputVector);
//     auto gradientFunction = cudaq::gradients::central_difference();
//     gradient = gradientFunction.compute(inputVector, terrain, resultAtX);

//     std::cout << "params: ";
//     for (auto param : inputVector)
//     {
//         std::cout << std::round(param * 100) / 100.00 << " , ";
//     }
//     std::cout << "\n";

//     std::cout << "gradient: ";
//     for (auto param : gradient)
//     {
//         std::cout << std::round(param * 100) / 100.00 << " , ";
//     }
//     std::cout << "\n";
//     std::cout << "Looped once: " << resultAtX << "\n";
//     std::cout << "\n\n\n\n";

//     return resultAtX;
// }