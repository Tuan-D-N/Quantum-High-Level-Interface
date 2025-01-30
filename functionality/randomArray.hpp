#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuComplex.h>

template<typename T>
std::vector<T> generateNormalizedRandomVectorState(int nQubits) {
    int size = 1 << nQubits; // 2^nQubits elements
    std::vector<T> vec(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float norm = 0.0f;

    // Generate random complex numbers and compute norm
    for (auto& v : vec) {
        float real = dist(gen);
        float imag = dist(gen);
        v = {real, imag};
        norm += real * real + imag * imag;
    }

    norm = std::sqrt(norm);

    // Normalize the vector
    for (auto& v : vec) {
        v.x /= norm;
        v.y /= norm;
    }

    return vec;
}

void generateRandomArray(double *arr, std::size_t size);    