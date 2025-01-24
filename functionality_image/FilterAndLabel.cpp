#include <vector>
#include "FilterAndLabel.hpp"

std::pair<std::vector<std::vector<float>>, std::vector<int>> filter_and_label(
    const std::vector<std::vector<float>> &images,
    const std::vector<int> &labels,
    const std::vector<int> &target_digits)
{
    std::vector<std::vector<float>> filtered_images;
    std::vector<int> filtered_labels;

    for (size_t i = 0; i < labels.size(); ++i)
    {
        auto it = std::find(target_digits.begin(), target_digits.end(), labels[i]);
        if (it != target_digits.end())
        {
            filtered_images.push_back(images[i]);
            filtered_labels.push_back(std::distance(target_digits.begin(), it)); // Renormalized to 0, 1, 2, ...
        }
    }

    return {filtered_images, filtered_labels};
}