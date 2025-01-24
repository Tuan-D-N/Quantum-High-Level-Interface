#pragma once
#include <vector>

/// @brief Filters out a list of images and labels for specific label results. Will renormalise result to from 1-N in order of target_digits
/// @param images The vector of images
/// @param labels The vector of labels
/// @param target_digits A vector of results you would like to filter.
/// @return {images, labels} a pair of filtered results
std::pair<std::vector<std::vector<float>>, std::vector<int>> filter_and_label(
    const std::vector<std::vector<float>> &images,
    const std::vector<int> &labels,
    const std::vector<int> &target_digits);
