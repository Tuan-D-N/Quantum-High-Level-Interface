#include <iostream>
#include "Runner/TestingInitSpeeds.hpp"
#include "CudaControl/Helper.hpp"
#include "functionality/ClockTimer.hpp"

int main() //Testing Inits
{
    const int numberOfQubits = 3;
    const bool function = true;
    {
        auto timer = Timer("Number of Qubits = " + std::to_string(numberOfQubits));
        if constexpr (function)
        {
            CHECK_BROAD_ERROR(initByFunction<numberOfQubits>());
        }
        else
        {
            CHECK_BROAD_ERROR(initByHand<numberOfQubits>());
        }
    }
    return 1;
}
