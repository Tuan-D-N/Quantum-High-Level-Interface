#pragma once
#include <vector>


/// @brief Generates a one hot code state of results. If given 3: {1} -> {1,0,0}; {2} -> {0,1,0}; {3} -> {0,0,1}
/// @param labels List of labels as integers
/// @param depth Number of possible integers
/// @return One hot code 2D vector of results.
std::vector<std::vector<float>> one_hot_encode(const std::vector<int> &labels, int depth);