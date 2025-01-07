#pragma once
#include <cuComplex.h>
bool almost_equal(cuDoubleComplex x, cuDoubleComplex y) {
    const double eps = 1.0e-5;
    const cuDoubleComplex diff = cuCsub(x, y);
    return (cuCabs(diff) < eps);
}

bool almost_equal(double x, double y) {
    const double eps = 1.0e-5;
    const double diff = x - y;
    return (abs(diff) < eps);
}