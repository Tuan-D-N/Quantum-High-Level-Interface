#include <span>
#include <vector>
#include <array>
#include <vector>
#include <cuComplex.h>
#include "../CuQuantumControl/Precision.hpp"
#include "../CudaControl/Helper.hpp"


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
