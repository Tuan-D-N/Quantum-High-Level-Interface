#include <vector>
#include <stdexcept>
#include <numeric>
#include <cuComplex.h>
#include "SquareNorm.hpp"

void square_normalize(std::vector<float> &state)
{
    float total = 0;
    for (size_t i = 0; i < state.size(); ++i)
    {
        total += state[i] * state[i];
    }

    // Normalize the squared elements
    if (total == 0.0f)
    {
        throw std::runtime_error("Cannot normalize a vector with a total of zero.");
    }
    for (float &val : state)
    {
        val /= total;
    }
}

void square_normalise_all(std::vector<std::vector<float>> &states)
{
    for (std::vector<float> &picture : states)
    {
        square_normalize(picture);
    }
}


bool square_normalize_omp(std::vector<cuDoubleComplex>& b) {
    if (b.empty()) {
        return true;
    }

    double squared_norm = 0.0;
    
    // #pragma omp parallel for: Creates a team of threads and distributes the loop iterations.
    // reduction(+:squared_norm): Ensures that the partial sums from each thread are safely combined.
    #pragma omp parallel for reduction(+:squared_norm)
    for (size_t i = 0; i < b.size(); ++i) {
        const auto& z = b[i];
        // |z|^2 = real(z)^2 + imag(z)^2
        squared_norm += cuCreal(z) * cuCreal(z) + cuCimag(z) * cuCimag(z);
    }

    double tolerance = 1e-15; 
    if (squared_norm < tolerance) {
        return false;
    }

    double norm = std::sqrt(squared_norm);
    
    double inverse_norm = 1.0 / norm;
    
    
    // #pragma omp parallel for: Distribute the division operation across threads.
    // The loop body modifies independent elements (b[i]), so no reduction is needed.
    #pragma omp parallel for
    for (size_t i = 0; i < b.size(); ++i) {
        b[i].x *= inverse_norm;
        b[i].y *= inverse_norm;        
    }

    return true;
}