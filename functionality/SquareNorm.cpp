#include <vector>
#include <stdexcept>
#include <numeric>
#include "SquareNorm.hpp"

void square_normalize(std::vector<float> &state)
{
    float total = 0;
    for (size_t i = 0; i < state.size(); ++i)
    {
        total += state[i] * state[i];
    }

    // Normalize the squared elements
    if (total == 0.0f)
    {
        throw std::runtime_error("Cannot normalize a vector with a total of zero.");
    }
    for (float &val : state)
    {
        val /= total;
    }
}

void square_normalise_all(std::vector<std::vector<float>> &states)
{
    for (std::vector<float> &picture : states)
    {
        square_normalize(picture);
    }
}
