#include <vector>
#include <stdexcept>
#include <numeric>

/// @brief Square normalises the state by dividing by sum(state^2)
/// @param state IN/OUT: state to normalise.
void square_normalize(std::vector<float> &state);

/// @brief Square normalises many states by dividing each state by sum(state^2). It is just a loop.
/// @param state IN/OUT: states to normalise.
void square_normalise_all(std::vector<std::vector<float>> &states);

/**
 * @brief Square-normalizes a vector of cuDoubleComplex in-place.
 * * Divides every element by the L2-norm (magnitude) of the vector.
 * The L2-norm squared is the sum of the squared magnitudes of the elements: ||state||^2 = sum(|state_i|^2).
 * * @param state The vector to be normalized. Modified in-place.
 * @return True if normalization was successful, false if the vector is a zero vector.
 */
bool square_normalize(std::vector<cuDoubleComplex>& state);