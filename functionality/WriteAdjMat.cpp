#include <cmath>
#include <tuple>
#include <vector>
#include "WriteAdjMat.hpp"
#include "Utilities.hpp"
#include "Linspace.hpp"
#include "OddRound.hpp"
#include <cassert>

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
    return 1 << evenQubits; // 2^evenqubits
}
int getColumnIndexSizeMini(int evenQubits)
{
    return 4 * getRowOffsetSizeMini(evenQubits); // max of 4 each row
}
int getValuesSizeMini(int evenQubits)
{
    return 4 * getRowOffsetSizeMini(evenQubits); // max of 4 each row
}

void writeMatAMiniCSR(int *rowOffset, int *columnIndex, complex *values, int evenQubits, int &rowOffsetSize, int &columnIndexSize, int &valuesSize)
{
    assert(isEven(evenQubits));

    int totalQubits = evenQubits + 1;        // We need 1 extra qubit to get both sides
    int halfMatLength = pow(2, evenQubits);  // At half length
    int fullMatLength = pow(2, totalQubits); // At full length

    int rLen = pow(2, evenQubits / 2);     // Length of each list
    int thetaLen = pow(2, evenQubits / 2); // Happens to be square grid so they are the same
    int xLen = pow(2, evenQubits / 2);
    int yLen = pow(2, evenQubits / 2);

    int maxR = rLen - 1; // RList = [-3 -1 1 3] so length is 4
    int maxX = maxR;
    int maxY = maxR;

    double xStep = (maxX - (-maxX)) / (xLen - 1); // endings inclusive scheme
    double yStep = (maxY - (-maxY)) / (xLen - 1); // endings inclusive scheme

    auto toRThetaIndex = [maxR, thetaLen, rLen](int rIndex, int thetaIndex)
    {
        return thetaIndex * rLen + rIndex; // r is fast, theta is slow
    };

    int ValueIter = 0;
    int RowIter = 0;
    int ColumnIter = 0;
    for (int i_x = 0; i_x < xLen; ++i_x)
    {
        double x = -maxX + xStep * i_x;
        for (int i_y = 0; i_y < yLen; ++i_y)
        {
            double y = -maxY + yStep * i_y;

            double r = sqrt(x * x + y * y);                     // Radial distance
            double theta = atan2(y, x);                         // angle
            double rIndex = (r + maxR) / 2.0;                   // index from 0 -> maxRLen-1, could be between
            double thetaIndex = (thetaLen - 1) * theta / 180.0; // index from 0 -> thetaLen-1, could be between

            int lrIndex = std::floor(rIndex);
            int urIndex = std::ceil(rIndex);

            int lthetaIndex = std::floor(thetaIndex);
            int uthetaIndex = std::ceil(thetaIndex);

            if (lrIndex == urIndex && lthetaIndex == uthetaIndex)
            {
                int index = toRThetaIndex(lrIndex, lthetaIndex); // Only 1 element

                rowOffset[RowIter] = ValueIter;
                values[ValueIter] = complex(1);
                columnIndex[ColumnIter] = index;

                ++RowIter;
                ++ValueIter;
                ++ColumnIter;
            }
            else if (lrIndex == urIndex)
            {
                int index1 = toRThetaIndex(lrIndex, lthetaIndex); // Lowest
                int index2 = toRThetaIndex(lrIndex, uthetaIndex); // Highest

                rowOffset[RowIter] = ValueIter;
                values[ValueIter] = complex(CorrelationHelper(lthetaIndex, thetaIndex, 1.0));
                columnIndex[ColumnIter] = index1;
                ++RowIter;
                ++ValueIter;
                ++ColumnIter;

                values[ValueIter] = complex(CorrelationHelper(uthetaIndex, thetaIndex, 1.0));
                columnIndex[ColumnIter] = index2;
                ++ValueIter;
                ++ColumnIter;
            }
            else if (lthetaIndex == uthetaIndex)
            {
                int index1 = toRThetaIndex(lrIndex, lthetaIndex); // Lowest
                int index2 = toRThetaIndex(urIndex, lthetaIndex); // Highest

                rowOffset[RowIter] = ValueIter;
                values[ValueIter] = complex(CorrelationHelper(lrIndex, rIndex, 1.0));
                columnIndex[ColumnIter] = index1;
                ++RowIter;
                ++ValueIter;
                ++ColumnIter;

                values[ValueIter] = complex(CorrelationHelper(urIndex, rIndex, 1.0));
                columnIndex[ColumnIter] = index2;
                ++ValueIter;
                ++ColumnIter;
            }
            else
            {
                int index1 = toRThetaIndex(lrIndex, lthetaIndex); // Lowest
                int index2 = toRThetaIndex(urIndex, lthetaIndex); // R changes faster so will have smaller index
                int index3 = toRThetaIndex(lrIndex, uthetaIndex); // Theta changes slower so will have larger index
                int index4 = toRThetaIndex(urIndex, uthetaIndex); // Highest

                rowOffset[RowIter] = ValueIter;
                values[ValueIter] = complex(CorrelationHelper(lrIndex, rIndex, 1.0) * CorrelationHelper(lthetaIndex, thetaIndex, 1.0));
                columnIndex[ColumnIter] = index1;
                ++RowIter;
                ++ValueIter;
                ++ColumnIter;

                values[ValueIter] = complex(CorrelationHelper(urIndex, rIndex, 1.0) * CorrelationHelper(lthetaIndex, thetaIndex, 1.0));
                columnIndex[ColumnIter] = index2;
                ++ValueIter;
                ++ColumnIter;

                values[ValueIter] = complex(CorrelationHelper(lrIndex, rIndex, 1.0) * CorrelationHelper(uthetaIndex, thetaIndex, 1.0));
                columnIndex[ColumnIter] = index3;
                ++ValueIter;
                ++ColumnIter;

                values[ValueIter] = complex(CorrelationHelper(urIndex, rIndex, 1.0) * CorrelationHelper(uthetaIndex, thetaIndex, 1.0));
                columnIndex[ColumnIter] = index4;
                ++ValueIter;
                ++ColumnIter;
            }
        }
    }
    rowOffsetSize = RowIter;
    columnIndexSize = ColumnIter;
    valuesSize = ValueIter;
}

std::vector<std::vector<double>> csrToDense(
    const cuDoubleComplex *values,  // Non-zero values
    const std::vector<int> &rowPtr, // Row pointers
    const std::vector<int> &cols,   // Column indices
    int rows,                       // Number of rows
    int colsCount                   // Number of columns
)
{
    // Initialize a dense matrix with zeros
    std::vector<std::vector<double>> dense(rows, std::vector<double>(colsCount, 0));

    // Iterate through each row
    for (int i = 0; i < rows; ++i)
    {
        // Non-zero elements for the row are in the range [rowPtr[i], rowPtr[i + 1])
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
        {
            dense[i][cols[j]] = cuCreal(values[j]);
        }
    }
    return dense;
}