#pragma once
#include <span>
#include <vector>
#include <array>
#include <vector>
#include <cuComplex.h>
#include "../CuQuantumControl/Precision.hpp"
#include "../CudaControl/Helper.hpp"


template <precision SelectPrecision = precision::bit_64>
std::pair<double, double>
measure1QubitUnified(std::span<PRECISION_TYPE_COMPLEX(SelectPrecision)> dataLocation);
