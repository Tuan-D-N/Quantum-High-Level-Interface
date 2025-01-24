#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <zlib.h>

/// @brief MAY CONTAIN UNDEFINED BEHAVIOUR. Read file of IDX format data pictures to vector of vector
/// @param filename names of file, relative path to executable
/// @param num_images the number of images in the file
/// @param rows the number of rows in each image
/// @param cols the number of columns in each image
/// @return Vector of images. Images are a vector of floats. Therefore returns a vector of vectors of floats
std::vector<std::vector<float>> read_images(const std::string &filename, int num_images, int rows, int cols);

/// @brief MAY CONTAIN UNDEFINED BEHAVIOUR. Read file of IDX format data labels to vector of ints. 
/// @param filename name of the file
/// @param num_labels the number of labels to read from file
/// @return 1D vector containing ints reading labels of integers    
std::vector<int> read_labels(const std::string &filename, int num_labels);

/// @brief Load MNIST data set
/// @param train_images_file path to training image file
/// @param train_labels_file path to training label file
/// @param test_images_file path to test image file
/// @param test_labels_file path to test label file
/// @param train_images_count number of training images in file
/// @param test_images_count number of test images in file
/// @param rows row size of image
/// @param cols column size of image
/// @return {{train_image, train_label},{test_image, test_label}}
std::pair<std::pair<std::vector<std::vector<float>>, std::vector<int>>,
          std::pair<std::vector<std::vector<float>>, std::vector<int>>>
load_mnist(const std::string &train_images_file = "train-images.idx3-ubyte",
           const std::string &train_labels_file = "train-labels.idx1-ubyte",
           const std::string &test_images_file = "t10k-images.idx3-ubyte",
           const std::string &test_labels_file = "t10k-labels.idx1-ubyte",
           const int train_images_count = 60000,
           const int test_images_count = 10000,
           const int rows = 28,
           const int cols = 28);