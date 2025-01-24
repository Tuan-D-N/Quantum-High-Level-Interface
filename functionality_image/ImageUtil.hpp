#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <zlib.h>

/// @brief Print image in ascii format
/// @param image vector storing image data
/// @param label label of the image (could leave as dummy)
/// @param rows number of rows in the image
/// @param cols number of cols in the image
void display_image(const std::vector<float> &image, const int label, int rows, int cols);


/// @brief Print many images in ascii format
/// @param images vector of (vectors, each storing images) 
/// @param label vectors of labels
/// @param rows number of rows in each image
/// @param cols number of columns in each image
/// @param num_images the number of images to print out (max: the size of vector of images/labels)
void display_images(const std::vector<std::vector<float>> &images, const std::vector<int> &label, int rows, int cols, int num_images);


/// @brief To normal pixel values from [0,255] to [0,1] and pad the image outsides
/// @param images in/out: vector of (vectors, each storing images) 
/// @param width number of rows in each image
/// @param height number of columns in each image
/// @param target_width target image width
/// @param target_height target image height
void normalize_and_pad(std::vector<std::vector<float>> &images, const int width, const int height, const int target_width = 32, const int target_height = 32);

