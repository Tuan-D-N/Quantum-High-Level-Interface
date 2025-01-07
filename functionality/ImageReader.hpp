#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>


std::vector<float> readImageAsNormalizedFloat(const std::string &imagePath, int flags = cv::IMREAD_GRAYSCALE);