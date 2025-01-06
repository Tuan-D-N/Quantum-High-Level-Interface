#pragma once
#include <vector>
#include <complex>



/// @brief Used helper in getMatA
/// @param a starting value
/// @param b ending value
/// @return If the distance a-b = 2 then return 0.
///
/// If the distance a-b = 0 then return 1
double getMatAHelperDiffCorrel(double a, double b);

/// @brief Generate the double interpolation matrix
/// @param evenQubits number of qubits. Have to be even.
/// @return flattened vector of a 2^(n+1)*2^(n+1) matrix
std::vector<std::vector<std::complex<double>>> getMatA(int evenQubits = 6);



/// @brief Generate the mini interpolation matrix
/// @param evenQubits number of qubits. Have to be even
/// @return flattened vector of a 2^(n)*2^(n) matrix
std::vector<std::vector<std::complex<double>>> getMatAMini(int evenQubits = 6);



/// @brief take the kroniker product of 2 column vectors to be 1 column vector
/// @tparam T type of slow vec
/// @tparam U type of fast vec
/// @param slowChange this vector changes once everytime all fast vectors have went throught all. The most signifcant digit.
/// @param fastChange this vector changes every new vector element. The least significant digit.
/// @param slowChangeFirst if true: puts slow vector first in the tuple list
/// @return A 1D vector of tuples containing every combination of vectors of the result.
template<typename T, typename U>
std::vector<std::tuple<T, U>> allPermuteOfVectors(std::vector<T> slowChange, std::vector<U> fastChange, bool slowChangeFirst = true); 
