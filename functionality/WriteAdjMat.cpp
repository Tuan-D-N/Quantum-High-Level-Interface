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

    double rStep = (maxR - minR) / (rLen - 1);                 // endings inclusive scheme
    double thetaStep = (maxTheta - minTheta) / (thetaLen - 1); // endings inclusive scheme

    int ValueIter = 0;
    int RowIter = 0;
    int ColumnIter = 0;

    for (int i_theta = 0; i_theta < thetaLen; ++i_theta)
    {
        double thetaValue = minTheta + thetaStep * i_theta;
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
                rowOffset[RowIter] = ValueIter;
                ++RowIter;

                int index1 = toXYIndex(lx, ly); // one value
                values[ValueIter] = complex(1);
                ++ValueIter;
                columnIndex[ColumnIter] = index1;
                ++ColumnIter;
            }
            else if (lx == ux)
            {
                rowOffset[RowIter] = ValueIter;
                ++RowIter;

                int index1 = toXYIndex(lx, ly); // smaller value first
                values[ValueIter] = complex(CorrelationHelper(ly, y));
                ++ValueIter;
                columnIndex[ColumnIter] = index1;
                ++ColumnIter;

                int index2 = toXYIndex(lx, uy); // larger value second
                values[ValueIter] = complex(CorrelationHelper(uy, y));
                ++ValueIter;
                columnIndex[ColumnIter] = index2;
                ++ColumnIter;
            }
            else if (ly == uy)
            {
                rowOffset[RowIter] = ValueIter;
                ++RowIter;

                int index1 = toXYIndex(lx, ly); // smaller value first
                values[ValueIter] = complex(CorrelationHelper(lx, x));
                ++ValueIter;
                columnIndex[ColumnIter] = index1;
                ++ColumnIter;

                int index2 = toXYIndex(ux, ly); // larger value second
                values[ValueIter] = complex(CorrelationHelper(ux, x));
                ++ValueIter;
                columnIndex[ColumnIter] = index2;
                ++ColumnIter;
            }
            else
            {
                rowOffset[RowIter] = ValueIter;
                ++RowIter;

                int index1 = toXYIndex(lx, ly); // smaller value first
                values[ValueIter] = complex(CorrelationHelper(lx, x) * CorrelationHelper(ly, y));
                ++ValueIter;
                columnIndex[ColumnIter] = index1;
                ++ColumnIter;

                int index2 = toXYIndex(lx, uy); // y changes faster than x
                values[ValueIter] = complex(CorrelationHelper(lx, x) * CorrelationHelper(uy, y));
                ++ValueIter;
                columnIndex[ColumnIter] = index1;
                ++ColumnIter;

                int index3 = toXYIndex(ux, ly); // x changes slower
                values[ValueIter] = complex(CorrelationHelper(ux, x) * CorrelationHelper(ly, y));
                ++ValueIter;
                columnIndex[ColumnIter] = index1;
                ++ColumnIter;

                int index4 = toXYIndex(ux, uy); // large value
                values[ValueIter] = complex(CorrelationHelper(ux, x) * CorrelationHelper(uy, y));
                ++ValueIter;
                columnIndex[ColumnIter] = index1;
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