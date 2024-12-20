#include <gtest/gtest.h>
#include <tuple>
#include "../functionality/GetAdjMat.hpp"
#include "../functionality/Linspace.hpp"



TEST(GetMat, getMatAHelperDiffCorrel1) {
    std::vector<double> a = { 0.31, -0.72, -0.74, 0.07, 0.47, -0.07, 0.14, -0.25, 0.7, 0.58 };
    std::vector<double> b = { 0.38, -0.7, -0.25, 0.4, 0.49, 0.78, 0.18, 0.26, 0.69, 0.14 };
    std::vector<double> result = { 0.965, 0.99, 0.755, 0.835, 0.99, 0.575, 0.98, 0.745, 0.995, 0.78 };

    ASSERT_TRUE(a.size() == b.size() && b.size() == result.size());

    for (int i = 0; i < a.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(getMatAHelperDiffCorrel(a[i], b[i]), result[i]);
    }
}

TEST(GetMat, getMatAHelperDiffCorrel2) {
    std::vector<double> a = { -1.06, -1.23, -2.89, -2.94, -1.48, -1.35, -1.15, -1.41, -2.48, -1.67 };
    std::vector<double> b = { -2.46, -1.37, -2.64, -2.02, -1.48, -2.73, -1.95, -2.61, -1.12, -2.53 };
    std::vector<double> result = { 0.3, 0.93, 0.875, 0.54, 1., 0.31, 0.6, 0.4, 0.32, 0.57 };

    ASSERT_TRUE(a.size() == b.size() && b.size() == result.size());

    for (int i = 0; i < a.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(getMatAHelperDiffCorrel(a[i], b[i]), result[i]);
    }
}

TEST(GetMat, getMatAHelperDiffCorrel3) {
    std::vector<double> a = { 1.7, 2.62, 2.34, 2.73, 2.53, 2.58, 1.49, 1.78, 1.9, 2.97 };
    std::vector<double> b = { 1.36, 2.59, 2., 2.06, 1.32, 2., 1.56, 2.02, 2.76, 1.47 };
    std::vector<double> result = { 0.83, 0.985, 0.83, 0.665, 0.395, 0.71, 0.965, 0.88, 0.57, 0.25 };

    ASSERT_TRUE(a.size() == b.size() && b.size() == result.size());

    for (int i = 0; i < a.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(getMatAHelperDiffCorrel(a[i], b[i]), result[i]);
    }
}


TEST(GetMat, permuteVectors)
{
    int evenQubits = 6;
    int rLen = pow(2, evenQubits / 2);
    int thetaLen = pow(2, evenQubits / 2);
    int xLen = pow(2, evenQubits / 2);
    int yLen = pow(2, evenQubits / 2);

    int maxR = rLen - 1;

    auto xList = linspaceVec(-maxR, maxR, xLen);
    auto yList = linspaceVec(-maxR, maxR, yLen);
    auto thetaList = linspaceVec(0.0, M_PI, thetaLen);
    auto rList = linspaceVec(-maxR, maxR, rLen);

    auto RThetaPermute = allPermuteOfVectors<double, double>(thetaList, rList, false);
    ASSERT_TRUE(thetaList.size() * rList.size() == RThetaPermute.size());
    auto iterRTheta = RThetaPermute.cbegin();
    for (auto thetaValue : thetaList)
    {
        for (auto rValue : rList)
        {
            auto currentTuple = (std::make_tuple(rValue, thetaValue));
            EXPECT_TRUE((*iterRTheta.base()) == currentTuple);
            ++iterRTheta;
        }
    }

    auto XYPermute = allPermuteOfVectors<double, double>(xList, yList, true); 
    auto iterXY = XYPermute.begin();
    for (auto xValue : xList)
    {
        for (auto yValue : yList)
        {
            auto currentTuple = (std::make_tuple(xValue, yValue));
            EXPECT_TRUE((*iterXY.base()) == currentTuple);
            ++iterXY;
        }
    }

}