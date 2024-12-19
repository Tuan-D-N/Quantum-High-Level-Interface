#include "Linspace.hpp"

#include <vector>


// Function to generate linearly spaced values from start to end
std::vector<double> linspaceVec(double start, double end, int n, bool includeEnd) {

    std::vector<double> result;

    // Calculate step size
    double step;

    if (includeEnd)
        step = (end - start) / (n - 1);
    else
        step = (end - start) / (n);

    // Generate linearly spaced values
    for (int i = 0; i < n; ++i) {
        result.push_back(start + i * step);
    }

    return result;
}