#pragma once
#include <span>
#include <vector>
#include <array>
#include <vector>
#include <cuComplex.h>
#include "../CuQuantumControl/Precision.hpp"
#include "../CudaControl/Helper.hpp"


template <precision SelectPrecision = precision::bit_64>
std::array<double, 2>
measure1QubitUnified(std::span<PRECISION_TYPE_COMPLEX(SelectPrecision)> dataLocation);
