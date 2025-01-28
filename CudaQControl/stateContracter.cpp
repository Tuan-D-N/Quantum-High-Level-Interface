#include <span>
#include <vector>
#include <array>
#include <vector>
#include <cuComplex.h>
#include <stdexcept>
#include "../CuQuantumControl/Precision.hpp"
#include "../CudaControl/Helper.hpp"
#include "stateContracter.hpp"

template <precision SelectPrecision>
std::pair<double, double>
measure1QubitUnified(std::span<PRECISION_TYPE_COMPLEX(SelectPrecision)> dataLocation)
{
    if (dataLocation.size % 2 != 0)
    {
        throw std::invalid_argument("measure1QubitUnified: Not even");
    }
    else if (dataLocation.size == 0)
    {
        throw std::invalid_argument("measure1QubitUnified: No input");
    }

    std::array<double, 2> results = {0, 0};

    for (int measureState = 0; measureState < 2; ++measureState)
    {
        int start = dataLocation.size() / 2 * measureState;
        int end = dataLocation.size() / 2 * (measureState + 1);
        for (int index = start; index < end; ++index)
        {
            results[measureState] += dataLocation[index].x * dataLocation[index].x +
                                     dataLocation[index].y * dataLocation[index].y;
        }
    }

    // Renormalise
    double total = results[0] + results[1];
    results[0] /= total;
    results[1] /= total;

    return std::pair<double, double>(results[0], results[1]);
}