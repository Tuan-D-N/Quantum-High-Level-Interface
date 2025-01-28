#pragma once
#include <span>
#include <vector>
#include <array>
#include <vector>
#include <cuComplex.h>
#include "../CuQuantumControl/Precision.hpp"
#include "../CudaControl/Helper.hpp"

template <precision SelectPrecision = precision::bit_64>
int GetStateProbability(int nQubits,
                        PRECISION_TYPE_COMPLEX(SelectPrecision) * d_sv,
                        std::span<const int> qubitsToCare,
                        std::vector<double> &out_AssociatedProbability);

template <precision SelectPrecision = precision::bit_64>
std::pair<double, double>
measure1QubitUnified(std::span<PRECISION_TYPE_COMPLEX(SelectPrecision)> dataLocation);
