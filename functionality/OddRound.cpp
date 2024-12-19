#include "OddRound.hpp"
#include <cmath>

double roundToHigherOdd(double num) {
    double roundedUp = std::ceil(num);
    return (static_cast<int>(roundedUp) % 2 == 0) ? roundedUp + 1 : roundedUp + 0;
}

double roundToLowerOdd(double num) {
    double roundedDown = std::floor(num);
    return (static_cast<int>(roundedDown) % 2 == 0) ? roundedDown - 1 : roundedDown - 0;
}
