#include <iostream>
#include <random>
#include "randomArray.hpp"

void generateRandomArray(double* arr, std::size_t size) {
    std::random_device rd; // Seed
    std::mt19937 gen(rd()); // Mersenne Twister RNG
    std::uniform_real_distribution<double> dist(0.0f, 1.0f); // Range [0, 1)

    for (std::size_t i = 0; i < size; ++i) {
        arr[i] = dist(gen);
    }
}