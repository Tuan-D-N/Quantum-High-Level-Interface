#include <iostream>
#include "Runner/GroverTest.hpp"
#include "CudaControl/Helper.hpp"

int main(int argc, char const *argv[])
{
    auto qubit_count = 1 < argc ? atoi(argv[1]) : 25;
    CHECK_BROAD_ERROR(grover(qubit_count));
    return 0;
}
