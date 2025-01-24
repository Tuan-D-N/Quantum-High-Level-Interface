#include <vector>
#include "OneHotCode.hpp"


std::vector<std::vector<float>> one_hot_encode(const std::vector<int> &labels, int depth)
{
    std::vector<std::vector<float>> one_hot_labels(labels.size(), std::vector<float>(depth, 0.0f));
    for (size_t i = 0; i < labels.size(); ++i)
    {
        one_hot_labels[i][labels[i]] = 1.0f;
    }
    return one_hot_labels;
}