#include <omp.h>
#include <iostream>
#include <chrono>

void sequential_matrix_multiply(int* A, int* B, int* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = 0;
            for (int k = 0; k < size; k++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

void parallel_matrix_multiply(int* A, int* B, int* C, int size) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = 0;
            for (int k = 0; k < size; k++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

int runOMPTest() {
    const int size = 1000;
    int* A = new int[size * size];
    int* B = new int[size * size];
    int* C = new int[size * size];
    
    // Initialize matrices A and B with some values
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i * size + j] = 1;
            B[i * size + j] = 1;
        }
    }

    // Sequential execution
    auto start = std::chrono::high_resolution_clock::now();
    sequential_matrix_multiply(A, B, C, size);
    auto end = std::chrono::high_resolution_clock::now();
    auto sequential_duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Sequential matrix multiplication time: " << sequential_duration << " seconds" << std::endl;

    // Parallel execution
    start = std::chrono::high_resolution_clock::now();
    parallel_matrix_multiply(A, B, C, size);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Parallel matrix multiplication time: " << parallel_duration << " seconds" << std::endl;

    // Calculate speedup
    double speedup = sequential_duration / parallel_duration;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
