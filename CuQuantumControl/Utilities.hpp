#pragma once
#include <cuComplex.h>

void set2ZeroState(cuDoubleComplex *d_sv, const int nSvSize);
void set2ZeroState(cuComplex *d_sv, const int nSvSize);


void set2NoState(cuDoubleComplex *d_sv, const int nSvSize);
void set2NoState(cuComplex *d_sv, const int nSvSize);