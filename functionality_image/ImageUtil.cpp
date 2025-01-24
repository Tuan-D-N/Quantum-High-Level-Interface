#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <zlib.h>
#include "ImageUtil.hpp"



void display_image(const std::vector<float> &image, const int label, int rows, int cols)
{
    std::cout << "Target: " << label << "\n";
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            float pixel = image[i * cols + j];
            char intensity = pixel > 0.5 ? '#' : (pixel > 0.25 ? '+' : (pixel > 0.125 ? '.' : ' '));
            std::cout << intensity;
        }
        std::cout << '\n';
    }
}

void display_images(const std::vector<std::vector<float>> &images, const std::vector<int> &label, int rows, int cols, int num_images)
{
    for (int i = 0; i < num_images; ++i)
    {
        std::cout << "Image " << i + 1 << ":\n";
        display_image(images[i], label[i], rows, cols);
        std::cout << "\n\n";
    }
}


void normalize_and_pad(std::vector<std::vector<float>> &images, const int width, const int height, const int target_width, const int target_height)
{
    for (auto &img : images)
    {
        // Normalize and pad (32x32 with padding of 2)
        std::vector<float> padded_img(target_width * target_height, 0.0f);
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                padded_img[(i + 2) * 32 + (j + 2)] = img[i * 28 + j] / 255.0f;
            }
        }
        img = padded_img;
    }
}

