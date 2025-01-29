#include "Utilities.hpp"
#include <cuComplex.h>

void set2ZeroState(cuDoubleComplex *d_sv, const int nSvSize)
{
    for (int i = 1; i < nSvSize; ++i)
    {
        d_sv[i] = make_cuDoubleComplex(0, 0);
    }
    d_sv[0] = make_cuDoubleComplex(1, 0);
}
void set2ZeroState(cuComplex *d_sv, const int nSvSize)
{
    for (int i = 1; i < nSvSize; ++i)
    {
        d_sv[i] = make_cuComplex(0, 0);
    }
    d_sv[0] = make_cuComplex(1, 0);
}

void set2NoState(cuDoubleComplex *d_sv, const int nSvSize)
{
    for (int i = 0; i < nSvSize; ++i)
    {
        d_sv[i] = make_cuDoubleComplex(0, 0);
    }
}
void set2NoState(cuComplex *d_sv, const int nSvSize)
{
    for (int i = 0; i < nSvSize; ++i)
    {
        d_sv[i] = make_cuComplex(0, 0);
    }
}

double magnitude(cuDoubleComplex complexNumber)
{
    return cuCabs(complexNumber);    
}

float magnitude(cuFloatComplex complexNumber)
{
    return cuCabsf(complexNumber);
}
