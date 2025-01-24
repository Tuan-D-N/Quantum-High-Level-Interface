#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <zlib.h>
#include "LoadImage.hpp"

// Function to read IDX format data
std::vector<std::vector<float>> read_images(const std::string &filename, int num_images, int rows, int cols)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Skip header (16 bytes)
    file.ignore(16);

    // Read image data
    std::vector<std::vector<float>> images(num_images, std::vector<float>(rows * cols));
    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < rows * cols; ++j)
        {
            unsigned char pixel;
            file.read(reinterpret_cast<char *>(&pixel), sizeof(pixel)); // Maybe UNDEFINED BEHAVIOUR
            images[i][j] = static_cast<float>(pixel);                   // Keep raw values for normalization later
        }
    }
    file.close();
    return images;
}

std::vector<int> read_labels(const std::string &filename, int num_labels)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Skip header (8 bytes)
    file.ignore(8);

    // Read label data
    std::vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; ++i)
    {
        unsigned char label;
        file.read(reinterpret_cast<char *>(&label), sizeof(label));
        labels[i] = static_cast<int>(label);
    }
    file.close();
    return labels;
}

// Load MNIST data
std::pair<std::pair<std::vector<std::vector<float>>, std::vector<int>>,
          std::pair<std::vector<std::vector<float>>, std::vector<int>>>
load_mnist(const std::string &train_images_file,
           const std::string &train_labels_file,
           const std::string &test_images_file,
           const std::string &test_labels_file,
           const int train_images_count,
           const int test_images_count,
           const int rows,
           const int cols

)
{
    // Read training data
    auto x_train = read_images(train_images_file, train_images_count, rows, cols);
    auto y_train = read_labels(train_labels_file, train_images_count);

    // Read test data
    auto x_test = read_images(test_images_file, test_images_count, rows, cols);
    auto y_test = read_labels(test_labels_file, test_images_count);

    return {{x_train, y_train}, {x_test, y_test}};
}