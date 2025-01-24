#include <vector>
#include <stdexcept>
#include <numeric>

/// @brief Square normalises the state by dividing by sum(state^2)
/// @param state IN/OUT: state to normalise.
void square_normalize(std::vector<float> &state);

/// @brief Square normalises many states by dividing each state by sum(state^2). It is just a loop.
/// @param state IN/OUT: states to normalise.
void square_normalise_all(std::vector<std::vector<float>> &states);