#pragma once

#include <vector>

/// @brief Creates a vector array of n numbers within a range
/// @param start starting value of vector
/// @param end ending value of vector
/// @param n number of numbers in the range
/// @param includeEnd 
/// @return vector of the linspaceVec
std::vector<double> linspaceVec(double start, double end, int n, bool includeEnd = true);


/// @brief Creates a vector array of n numbers within a range
/// @param start starting value of vector
/// @param end ending value of vector
/// @param n number of numbers in the range
/// @param includeEnd 
/// @return vector of the linspaceVec
std::vector<double> linspaceVec(int start, int end, int n, bool includeEnd = true);