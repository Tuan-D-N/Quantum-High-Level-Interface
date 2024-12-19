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

// Function to generate linearly spaced values from start to end
std::vector<double> linspaceVec(int start, int end, int n, bool includeEnd) {

    double start_p = static_cast<double>(start);
    double end_p = static_cast<double>(end);

    return linspaceVec(start_p, end_p, n, includeEnd);
}