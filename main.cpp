#include <iostream>
#include "Runner/CircuitTest.hpp"
#include "Runner/ApplyIRadon.hpp"
#include "CudaControl/Helper.hpp"
#include <cmath>

int main()
{
    for (int n = 1; n < 30; ++n)
    {
        CHECK_BROAD_ERROR(runSys3(n));
    }
    return 1;
}
