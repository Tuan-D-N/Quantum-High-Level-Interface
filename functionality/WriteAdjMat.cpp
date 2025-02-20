#include <cmath>
#include <tuple>
#include <vector>
#include "WriteAdjMat.hpp"
#include "Utilities.hpp"
#include "Linspace.hpp"
#include <functional>
#include "OddRound.hpp"
#include <cassert>
#include <iostream>

using complex = cuDoubleComplex;

double CorrelationHelper(double a, double b, double maxDifference = 2.0)
{
    return 1 - std::abs((a - b) / maxDifference);
}

template <typename T, typename U>
std::vector<std::tuple<T, U>> allPermuteOfVectors(std::vector<T> slowChange, std::vector<U> fastChange, bool slowChangeFirst)
{
    int length = slowChange.size() * fastChange.size();
    std::vector<std::tuple<double, double>> meshState(length);

    if (slowChangeFirst)
    {
        for (int i = 0; i < slowChange.size(); ++i)
        {
            for (int j = 0; j < fastChange.size(); ++j) // Can be parallelised
            {
                int index = i * fastChange.size() + j;
                meshState[index] = std::make_tuple(slowChange[i], fastChange[j]);
            }
        }
    }
    else
    {
        for (int i = 0; i < slowChange.size(); ++i)
        {
            for (int j = 0; j < fastChange.size(); ++j) // Can be parallelised
            {
                int index = i * fastChange.size() + j;
                meshState[index] = std::make_tuple(fastChange[j], slowChange[i]);
            }
        }
    }
    return meshState;
}

template std::vector<std::tuple<double, double>> allPermuteOfVectors<double, double>(std::vector<double>, std::vector<double>, bool);

int getRowOffsetSizeMini(int evenQubits)
{
    return (1 << evenQubits) + 1; // 2^evenqubits and 1 for end wrapping
}
int getColumnIndexSizeMini(int evenQubits)
{
    return 4 * getRowOffsetSizeMini(evenQubits); // max of 4 each row
}
int getValuesSizeMini(int evenQubits)
{
    return 4 * getRowOffsetSizeMini(evenQubits); // max of 4 each row
}

void writeMatAMiniCSC(int *ColumnOffset, int *rowIndex, complex *values, int evenQubits, int &ColumnOffsetSize, int &rowIndexSize, int &valuesSize, bool mask)
{
    assert(isEven(evenQubits));

    const int totalQubits = evenQubits + 1;        // We need 1 extra qubit to get both sides
    const int halfMatLength = pow(2, evenQubits);  // At half length
    const int fullMatLength = pow(2, totalQubits); // At full length

    const int rLen = pow(2, evenQubits / 2);     // Length of each list
    const int thetaLen = pow(2, evenQubits / 2); // Happens to be square grid so they are the same
    const int xLen = pow(2, evenQubits / 2);
    const int yLen = pow(2, evenQubits / 2);

    const int maxR = rLen - 1; // RList = [-3 -1 1 3] so length is 4
    const int minR = -maxR;
    const int maxTheta = 180;
    const int minTheta = 0;

    auto toXYIndex = [xLen, maxR](int x, int y)
    {
        assert(isOdd(x));
        assert(isOdd(y));
        x = (x + maxR) / 2;
        y = (y + maxR) / 2;
        return xLen * x + y;
    };

    double rStep = (maxR - minR) / static_cast<double>((rLen - 1));                 // endings inclusive scheme
    double thetaStep = (maxTheta - minTheta) / static_cast<double>((thetaLen - 1)); // endings inclusive scheme

    int ValueIter = 0;
    int ColumnOffsetIter = 0;
    int RowIter = 0;

    double lowerValue = 0;
    double upperValue = 1;
    double maxDistance = maxR;

    std::function<double(double)> maskFunc;

    if (mask)
    {
        maskFunc = [lowerValue, upperValue, maxDistance](double distance)
        {
            return lowerValue + (upperValue - lowerValue) * (abs(distance) / maxDistance);
        };
    }
    else
    {
        maskFunc = [](double distance)
        {
            return 1;
        };
    }

    for (int i_theta = 0; i_theta < thetaLen; ++i_theta)
    {
        double thetaValue = (minTheta + thetaStep * i_theta) * M_PI / 180;
        for (int i_r = 0; i_r < rLen; ++i_r)
        {
            double rValue = minR + rStep * i_r;

            double x = rValue * cos(thetaValue);
            double y = rValue * sin(thetaValue);

            double lx = roundToLowerOdd(x);
            double ux = roundToHigherOdd(x);
            double ly = roundToLowerOdd(y);
            double uy = roundToHigherOdd(y);
            if (lx == ux && ly == uy)
            {
                ColumnOffset[ColumnOffsetIter] = ValueIter;
                ++ColumnOffsetIter;

                int index1 = toXYIndex(lx, ly); // one value
                values[ValueIter] = complex(1 * maskFunc(rValue));
                ++ValueIter;
                rowIndex[RowIter] = index1;
                ++RowIter;
            }
            else if (lx == ux)
            {
                ColumnOffset[ColumnOffsetIter] = ValueIter;
                ++ColumnOffsetIter;

                int index1 = toXYIndex(lx, ly); // smaller value first
                values[ValueIter] = complex(CorrelationHelper(ly, y) * maskFunc(rValue));
                ++ValueIter;
                rowIndex[RowIter] = index1;
                ++RowIter;

                int index2 = toXYIndex(lx, uy); // larger value second
                values[ValueIter] = complex(CorrelationHelper(uy, y) * maskFunc(rValue));
                ++ValueIter;
                rowIndex[RowIter] = index2;
                ++RowIter;
            }
            else if (ly == uy)
            {
                ColumnOffset[ColumnOffsetIter] = ValueIter;
                ++ColumnOffsetIter;

                int index1 = toXYIndex(lx, ly); // smaller value first
                values[ValueIter] = complex(CorrelationHelper(lx, x) * maskFunc(rValue));
                ++ValueIter;
                rowIndex[RowIter] = index1;
                ++RowIter;

                int index2 = toXYIndex(ux, ly); // larger value second
                values[ValueIter] = complex(CorrelationHelper(ux, x) * maskFunc(rValue));
                ++ValueIter;
                rowIndex[RowIter] = index2;
                ++RowIter;
            }
            else
            {
                ColumnOffset[ColumnOffsetIter] = ValueIter;
                ++ColumnOffsetIter;

                int index1 = toXYIndex(lx, ly); // smaller value first
                values[ValueIter] = complex(CorrelationHelper(lx, x) * CorrelationHelper(ly, y) * maskFunc(rValue));
                ++ValueIter;
                rowIndex[RowIter] = index1;
                ++RowIter;

                int index2 = toXYIndex(lx, uy); // y changes faster than x
                values[ValueIter] = complex(CorrelationHelper(lx, x) * CorrelationHelper(uy, y) * maskFunc(rValue));
                ++ValueIter;
                rowIndex[RowIter] = index2;
                ++RowIter;

                int index3 = toXYIndex(ux, ly); // x changes slower
                values[ValueIter] = complex(CorrelationHelper(ux, x) * CorrelationHelper(ly, y) * maskFunc(rValue));
                ++ValueIter;
                rowIndex[RowIter] = index3;
                ++RowIter;

                int index4 = toXYIndex(ux, uy); // large value
                values[ValueIter] = complex(CorrelationHelper(ux, x) * CorrelationHelper(uy, y) * maskFunc(rValue));
                ++ValueIter;
                rowIndex[RowIter] = index4;
                ++RowIter;
            }
        }
    }
    ColumnOffset[ColumnOffsetIter] = ValueIter; // Closing offset
    ++ColumnOffsetIter;

    ColumnOffsetSize = ColumnOffsetIter;
    rowIndexSize = RowIter;
    valuesSize = ValueIter;
}
