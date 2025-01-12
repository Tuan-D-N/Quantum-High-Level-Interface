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