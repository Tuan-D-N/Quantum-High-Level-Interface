#include <iostream>
#include <gtest/gtest.h>
#include <cuComplex.h>
#include "../functionality/WriteAdjMat.hpp"
#include "../functionality/GetAdjMat.hpp"
#include "../functionality/Utilities.hpp"
#include <vector>

bool testQubitCompare(int evenqubits)
{
    int len = 1 << evenqubits;
    auto rowOffset = std::vector<int>(getRowOffsetSizeMini(evenqubits), 0);
    auto columnIndices = std::vector<int>(getColumnIndexSizeMini(evenqubits), 0);
    cuDoubleComplex *values = new cuDoubleComplex[getValuesSizeMini(evenqubits)];

    int rowOffsetSize;
    int columnIndicesSize;
    int valuesSize;

    writeMatAMiniCSC(rowOffset.data(),
                     columnIndices.data(),
                     values,
                     evenqubits,
                     rowOffsetSize,
                     columnIndicesSize,
                     valuesSize);

    auto matA = getMatAMini(evenqubits);

    auto matB = cscToDense(values, rowOffset, columnIndices, len, len);

    for (int i = 0; i < len; ++i)
    {
        for (int j = 0; j < len; ++j)
        {
            if(matA[i][j].real() - matB[i][j] > 0.003){
                std::cout << "failed " << i << "," << j << " " << matA[i][j].real() << " , " << matB[i][j] << "\n";
                return false;
            }
        }
    }
    delete[] values;
    return true;
}

TEST(WriteAdjMat, comparing2dense4) 
{
    EXPECT_TRUE(testQubitCompare(4));
}
TEST(WriteAdjMat, comparing2dense6) 
{
    EXPECT_TRUE(testQubitCompare(6));
}
TEST(WriteAdjMat, comparing2dense8) 
{
    EXPECT_TRUE(testQubitCompare(8));
}
TEST(WriteAdjMat, comparing2dense10) 
{
    EXPECT_TRUE(testQubitCompare(10));
}
TEST(WriteAdjMat, comparing2dense12) 
{
    EXPECT_TRUE(testQubitCompare(12));
}
TEST(WriteAdjMat, comparing2dense14) 
{
    EXPECT_TRUE(testQubitCompare(14));
}