#include <complex>
#include <vector>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuComplex.h>
#include <cstdlib>
#include <cstdint>
#include <device_launch_parameters.h>
#include <iostream>
#include "functionality/GetAdjMat.hpp"
#include "functionality/Utilities.hpp"


#ifdef __CUDACC__
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#else
#define a(...) 
#endif


void abc(){

}

int main(int argc, char const *argv[])
{
    int c,d;
    abc a((int)c,d) ();
    std::vector<std::vector<std::complex<double>>> matA = getMatA(12);
    print2DVector(matA);
    return 0;
}
