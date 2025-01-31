#include <omp.h>
#include <iostream>
#include <chrono>

void sequential_sum(int* arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
}

void parallel_sum(int* arr, int size) {
    int sum = 0;
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
}

int runOMPTest() {
    const int size = 10000000;
    int* arr = new int[size];
    
    // Initialize array with values
    for (int i = 0; i < size; i++) {
        arr[i] = 1; // Simple values to sum up
    }

    // Sequential execution
    auto start = std::chrono::high_resolution_clock::now();
    sequential_sum(arr, size);
    auto end = std::chrono::high_resolution_clock::now();
    auto sequential_duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Sequential time: " << sequential_duration << " seconds" << std::endl;

    // Parallel execution
    start = std::chrono::high_resolution_clock::now();
    parallel_sum(arr, size);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Parallel time: " << parallel_duration << " seconds" << std::endl;

    // Calculate speedup
    double speedup = sequential_duration / parallel_duration;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    delete[] arr;
    return 0;
}
