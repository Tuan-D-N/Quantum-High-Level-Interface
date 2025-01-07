#include "Utilities.hpp"
#include <vector>
#include <iostream>
#include <complex>

bool isEven(int num)
{
    return num % 2 == 0;
}

bool isOdd(int num)
{
    return num % 2 != 0;
}

template <Streamable T>
void print2DVector(const std::vector<std::vector<T>> &vec)
{
    for (const auto &row : vec)
    { // Iterate over each row
        for (T val : row)
        {                            // Iterate over each column in the row
            std::cout << val << " "; // Print the element
        }
        std::cout << "\n"; // Move to the next line after printing a row
    }
}

template void print2DVector<double>(const std::vector<std::vector<double>> &vec);
template void print2DVector<int>(const std::vector<std::vector<int>> &vec);
template void print2DVector<float>(const std::vector<std::vector<float>> &vec);
template void print2DVector<std::complex<double>>(const std::vector<std::vector<std::complex<double>>> &vec);
template void print2DVector<std::complex<int>>(const std::vector<std::vector<std::complex<int>>> &vec);
template void print2DVector<std::complex<float>>(const std::vector<std::vector<std::complex<float>>> &vec);

template <Streamable T>
void printVector(const std::vector<T> &vec)
{
    for (T val : vec)
    { // Iterate over the vector and print each element
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

template void printVector<double>(const std::vector<double> &vec);
template void printVector<int>(const std::vector<int> &vec);
template void printVector<float>(const std::vector<float> &vec);
template void printVector<std::complex<double>>(const std::vector<std::complex<double>> &vec);
template void printVector<std::complex<int>>(const std::vector<std::complex<int>> &vec);
template void printVector<std::complex<float>>(const std::vector<std::complex<float>> &vec);

