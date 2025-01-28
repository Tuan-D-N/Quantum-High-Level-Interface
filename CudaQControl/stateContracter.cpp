#include "stateContracter.hpp"

// TODO: FIX LATER
// TODO: NO IDEA HAVE THE QUBIT MEASURED LISTS WAS WRITTEN
template <precision SelectPrecision>
int GetStateProbability(int nQubits,
                        PRECISION_TYPE_COMPLEX(SelectPrecision) * d_sv,
                        std::span<const int> qubitsToCare, //{0,1}
                        std::vector<double> &out_AssociatedProbability)
{
    custatevecHandle_t handle = NULL;
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    CHECK_CUSTATEVECTOR(custatevecCreate(&handle));

    const int subStateQubitsSize = nQubits - qubitsToCare.size();
    const int measuredStateQubitsSize = qubitsToCare.size();

    std::vector<PRECISION_TYPE_COMPLEX(SelectPrecision)> outBuffer;

    std::vector<int> bitOrdering(subStateQubitsSize);
    // Iterate over the range 1 to n, adding numbers that are not in qubitsToCare
    for (int i = 1; i <= nQubits; ++i)
    {
        if (std::find(qubitsToCare.begin(), qubitsToCare.end(), i) == qubitsToCare.end())
        {
            bitOrdering.push_back(i);
        }
    }

    auto maskOrdering = qubitsToCare;

    for (int i = 0; i < (1 << measuredStateQubitsSize); ++i) // Loop through all possible
    {
        std::vector<int> maskBitString(measuredStateQubitsSize);

        // Fill maskBitString with the binary representation of i
        for (int bit = 0; bit < measuredStateQubitsSize; ++bit)
        {
            maskBitString[bit] = (i >> (measuredStateQubitsSize - 1 - bit)) & 1; // Extract each bit starting from the most significant
        }

        applyAccessorGet(handle,
                         nQubits,
                         bitOrdering,
                         maskBitString,
                         maskOrdering,
                         d_sv,
                         outBuffer,
                         extraWorkspace,
                         extraWorkspaceSizeInBytes);
    }

    if (extraWorkspace != nullptr)
    {
        std::cout << "Wordspace was freed\n";
        CHECK_CUDA(cudaFree(extraWorkspace));
    }
    CHECK_CUSTATEVECTOR(custatevecDestroy(handle));
}

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