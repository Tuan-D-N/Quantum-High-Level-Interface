#include <opencv2/opencv.hpp> // NOLINT
#include <vector>
#include <iostream>
#include "ImageReader.hpp"

std::vector<float> readImageAsNormalizedFloat(const std::string& imagePath, int flags = cv::IMREAD_GRAYSCALE) {
    // Read the image
    cv::Mat image = cv::imread(imagePath, flags);
    if (image.empty()) {
        throw std::runtime_error("Error: Could not open or find the image!");
    }

    // Convert image to float and normalize to [0, 1]
    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32F, 1.0 / 255.0);

    // Flatten the normalized float image into a single array
    return std::vector<float>(imageFloat.begin<float>(), imageFloat.end<float>());
}
