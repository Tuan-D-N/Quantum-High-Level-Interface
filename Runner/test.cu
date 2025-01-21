
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include "../functionality/WriteAdjMat.hpp"
#include "../functionality/ReadCsv.hpp"
#include "../functionality/Utilities.hpp"
#include "../functionality/fftShift.hpp"
#include "../functionality/Transpose.hpp"
#include "../CuQuantumControl/QftStateVec.hpp"
#include "../CudaControl/Helper.hpp"
#include "../CuSparseControl/ApplyMatrixA.hpp"
#include <cuComplex.h>
#include <iostream>
#include "ApplyIRadon.hpp"
#include <string>
#include <cassert>
#include <span>
#include <iostream>

void printSpan(std::span<const int> span)
{
    for (int value : span)
    {
        std::cout << value << " ";
    }
    std::cout << "\n";
}

void printSpan(int span)
{
    std::cout << "\n";
}

int main()
{
    printSpan(std::array<int,5>{1, 2, 3, 4, 5}); // Compiler deduces the size as 5

    return 0;
}