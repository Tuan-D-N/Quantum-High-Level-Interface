#pragma once
#include <vector>
#include <complex>
#include <cuComplex.h>


/// @brief The function speaks for itself
/// @param num integer only
/// @return true/false
bool isOdd(int num);



/// @brief Are you literally dumb to read this
/// @param num integer only
/// @return true/false
bool isEven(int num);


// Define a concept for types that can be used with std::ostream
template <typename T>
concept Streamable = requires(std::ostream& os, T value) {
    { os << value } -> std::same_as<std::ostream&>; // Ensures os << value is valid and returns std::ostream&
};


/// @brief Print a 2D vector out 
/// @tparam T a streamable type
/// @param vec input vector
template<Streamable T> void print2DVector(const std::vector<std::vector<T>>& vec);

/// @brief print a 1D vector out
/// @tparam T a streamable type
/// @param vec input vector
template<Streamable T> void printVector(const std::vector<T>& vec);

