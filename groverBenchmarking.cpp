#include <iostream>
#include "Runner/GroverTest.hpp"
#include "CudaControl/Helper.hpp"

int main(int argc, char const *argv[])
{
    CHECK_BROAD_ERROR(grover3<10>());
    return 0;
}
