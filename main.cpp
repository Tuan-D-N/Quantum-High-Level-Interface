#include <iostream>
#include "Runner/CircuitTest.hpp"
#include "CudaControl/Helper.hpp"

int main()
{
    // runner();
    // runSys2();

    CHECK_BROAD_ERROR(grover(3));
    return 1;
}
